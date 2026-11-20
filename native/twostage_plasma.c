/* Adapted from PLASMA compute/zheevd.c and pzhbtrd_static.c (double precision).
 * Copyright (C) 2008-2016 University of Tennessee, Innovative Computing Lab.
 * BSD-3-Clause; full notice in third_party/PLASMA_LICENSE.txt.
 *
 * Only the orthogonal reduction is used: NO eigenproblem is solved. The rows
 * [b^T; X] are multiplied by Q1 then Q2, preserving both transformations.
 */
#include "plasma.h"
#include "plasma_internal.h"
#include "plasma_core_blas.h"
#include "bulge.h"
#include <omp.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <sched.h>

// Wavefront PLASMA: il segnale release pubblica matrice e riflettori;
// il vicino usa acquire prima di leggerli, necessario anche su ARM.
static void wait_for(_Atomic int* progress,int task,int sweep) {
    while(atomic_load_explicit(progress+task,memory_order_acquire)!=sweep)
        sched_yield();
}

static int fed2rc_bulge(int n,int nb,int vb,double* A,int lda,double* V,double* tau,
                 double* D,double* E,plasma_workspace_t work) {
    const int nt=(n+nb-1)/nb,p=nt<omp_get_max_threads()?nt:omp_get_max_threads();
    const int shift=3,psize=2*nt+shift-1;
    _Atomic int* progress=malloc((size_t)psize*sizeof(*progress));
    if(!progress) return PlasmaErrorOutOfMemory;
    for(int i=0;i<psize;++i) atomic_init(progress+i,0);
    #pragma omp parallel num_threads(p)
    {
        const int tid=omp_get_thread_num();
        double* scratch=(double*)work.spaces[tid];
        int sweep_begin=0;
        for(int i=0;i<n-1;++i) for(int k=0;k<shift;++k)
            for(int sweep=sweep_begin;sweep<=i;++sweep) {
                const int task=(i-sweep)*shift+k;
                const int type=task==0?1:3-task%2;
                const int first=(task/2)*nb+sweep+1;
                const int last=first+nb-1<n-1?first+nb-1:n-1;
                const int owner=(first/nb)%p;
                const int done=(last>=n-1)||(type==2 && last>=n-2);
                if(tid==owner) {
                    if(task) wait_for(progress,task-1,sweep+1);
                    wait_for(progress,task+shift-1,sweep);
                    if(type==1)
                        plasma_core_dsbtrd_type1(n,nb,A,lda,V,tau,first,last,sweep,vb,2,scratch);
                    else if(type==2)
                        plasma_core_dsbtrd_type2(n,nb,A,lda,V,tau,first,last,sweep,vb,2,scratch);
                    else
                        plasma_core_dsbtrd_type3(n,nb,A,lda,V,tau,first,last,sweep,vb,2,scratch);
                    // Publishes ALL matrix, V and tau writes before the flag.
                    atomic_store_explicit(progress+task,sweep+1,memory_order_release);
                    if(done) for(int t=1;t<shift;++t)
                        atomic_store_explicit(progress+task+t,sweep+1,memory_order_release);
                }
                if(done) ++sweep_begin;
            }
    }
    free(progress);
    for(int j=0;j<n-1;++j) { D[j]=A[(size_t)j*lda]; E[j]=A[(size_t)j*lda+1]; }
    D[n-1]=A[(size_t)(n-1)*lda];
    return PlasmaSuccess;
}


/* Upstream pdormqr_blgtrd's WANTZ!=1 path unconditionally selects the LEFT
 * traversal ("if (versionL == 113)", the side check is commented out). A right
 * application needs the reverse order of those blocks. Use the same compact
 * reflectors and BLAS kernel, but traverse in the correct order. Independent
 * row tiles are distributed without any overlapping writes or nested teams.
 */
static int apply_q2_right(int n,int nb,int nr,int vb,double* V,double* T,
                            double* E,plasma_workspace_t work)
{
    const int nt=plasma_ceildiv(n-1,vb),ldv=nb+vb-1;
    const int chunk=imin(64,plasma_ceildiv(nr,omp_get_max_threads()));
    int active=0;
    #pragma omp parallel reduction(+:active)
    {
    int worked=0;
    #pragma omp for schedule(static)
    for(int row=0;row<nr;row+=chunk) {
        worked=1;
        const int len=imin(chunk,nr-row);
        double* scratch=(double*)work.spaces[omp_get_thread_num()];
        for(int bj=0;bj<nt;++bj) {
            const int first=bj*vb+1;
            const int mt=plasma_ceildiv(n-first-(bj==nt-1?0:1),nb);
            for(int bi=1;bi<=mt;++bi) {
                const int myrow=first+(mt-bi)*nb;
                const int vm=imin(nb+vb-1,n-myrow);
                const int vn=imin(vb,vm-((bj==nt-1 && bi==mt)?0:1));
                int vp,tp,taup,id;
                findVTpos(n,nb,vb,bj*vb,myrow,&vp,&taup,&tp,&id);
                if(vm>0 && vn>0)
                    plasma_core_dlarfb_gemm(PlasmaRight,PlasmaNoTrans,
                        PlasmaForward,PlasmaColumnwise,len,vm,vn,V+vp,ldv,
                        T+tp,vb,E+row+(size_t)myrow*nr,nr,scratch,len);
            }
        }
    }
    active+=worked;
    }
    return active;
}

int fed2rc_two_stage_reduce(const double *input, int n, const double *rows,
                            int nr, int tile, int ib, double *diagonal,
                            double *off, double *output, int *active_workers)
{
    plasma_desc_t A, T, R;
    plasma_workspace_t work;
    plasma_sequence_t sequence;
    plasma_request_t request;
    int have_a=0, have_t=0, have_r=0, have_work=0, code=0;
    const int nb = tile<n ? tile:n;
    const int ldab=2*nb+1, vb=imax(1,nb/4), ldt=vb;
    int blocks=0, ldv=0;
    double *band=NULL, *tau=NULL, *vectors=NULL, *triangular=NULL;
    plasma_sequence_init(&sequence); plasma_request_init(&request);
    code=plasma_desc_general_create(PlasmaRealDouble,tile,tile,n,n,0,0,n,n,&A);
    if (code) goto cleanup; have_a=1;
    code=plasma_descT_create(A,ib,PlasmaFlatHouseholder,&T);
    if (code) goto cleanup; have_t=1;
    code=plasma_desc_general_create(PlasmaRealDouble,tile,tile,nr,n,0,0,nr,n,&R);
    if (code) goto cleanup; have_r=1;
    code=plasma_workspace_create(&work,(size_t)ib*tile+4*(size_t)tile*tile,PlasmaRealDouble);
    if (code) goto cleanup; have_work=1;
    band=calloc((size_t)ldab*n,sizeof(double));
    findVTsiz(n,nb,vb,&blocks,&ldv);
    tau=calloc((size_t)blocks*vb,sizeof(double));
    vectors=calloc((size_t)ldv*blocks*vb,sizeof(double));
    triangular=calloc((size_t)ldt*blocks*vb,sizeof(double));
    if (!band || !tau || !vectors || !triangular) { code=PlasmaErrorOutOfMemory; goto cleanup; }
    #pragma omp parallel
    #pragma omp master
    {
        plasma_pdge2desc((double*)input,n,A,&sequence,&request);
        plasma_pdge2desc((double*)rows,nr,R,&sequence,&request);
    }
    #pragma omp parallel
    #pragma omp master
    {
        plasma_pdsy2sb(PlasmaLower,A,T,work,&sequence,&request);
        plasma_pdsycpy_tile2lapack_band(PlasmaLower,A,band,ldab,&sequence,&request);
    }
    // Wavefront with explicit release/acquire for weakly ordered CPUs (ARM).
    code=fed2rc_bulge(n,nb,vb,band,ldab,vectors,tau,diagonal,off,work);
    if(code) goto cleanup;
    #pragma omp parallel
    #pragma omp master
    {
        if (tile<n)
            plasma_pdormqr(PlasmaRight,PlasmaNoTrans,
                plasma_desc_view(A,tile,0,n-tile,n-tile),
                plasma_desc_view(T,T.mb,0,T.m-T.mb,T.n-T.nb),
                plasma_desc_view(R,0,tile,nr,n-tile),work,&sequence,&request);
        plasma_pddesc2ge(R,output,nr,&sequence,&request);
    }
    #pragma omp parallel
    {
        plasma_pdlarft_blgtrd(n,nb,vb,vectors,triangular,tau,&sequence,&request);
    }
    if(sequence.status==PlasmaSuccess)
        *active_workers=apply_q2_right(n,nb,nr,vb,vectors,triangular,output,work);
    code=sequence.status;
cleanup:
    free(triangular); free(vectors); free(tau); free(band);
    if (have_work) plasma_workspace_destroy(&work);
    if (have_r) plasma_desc_destroy(&R);
    if (have_t) plasma_desc_destroy(&T);
    if (have_a) plasma_desc_destroy(&A);
    return code;
}
