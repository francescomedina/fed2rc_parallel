// Optional experimental PLASMA backend. The default extension does not link it.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>
extern "C" {
#include "plasma.h"
int fed2rc_two_stage_reduce(const double*,int,const double*,int,int,int,
                           double*,double*,double*,int*);
}
namespace py=pybind11;
template<class T> using CArray=py::array_t<T,py::array::c_style>;
template<class T> using FArray=py::array_t<T,py::array::f_style>;
static std::mutex context_mutex;
static void require(bool ok,const char* msg) { if(!ok) throw py::value_error(msg); }
template<class T,int Flags> static void finite(const py::array_t<T,Flags>& a) {
    require(reinterpret_cast<std::uintptr_t>(a.data())%alignof(T)==0,"arrays must be aligned");
    for(py::ssize_t i=0;i<a.size();++i) require(std::isfinite(a.data()[i]),"inputs must be finite");
}
static void validate(const FArray<double>& A,const FArray<double>& b,const CArray<double>& X,
                     int threads,int tile,int ib) {
    require(A.ndim()==2 && A.shape(0)>0 && A.shape(0)==A.shape(1) && A.shape(0)<=INT_MAX,
            "A must be a nonempty square matrix");
    require(b.ndim()==2 && b.shape(0)==A.shape(0) && b.shape(1)>0 && b.shape(1)<=INT_MAX-2,
            "invalid b shape");
    require(X.ndim()==2 && X.shape(0)>0 && X.shape(1)==A.shape(0)
            && X.shape(0)<=INT_MAX-b.shape(1),"invalid X shape");
    require(threads>0,"threads must be positive");
    require(tile>=16 && tile%4==0 && tile<=4096,"bandwidth must be a multiple of 4 in [16,4096]");
    require(ib>0 && ib<=tile,"inner_block must be in [1,bandwidth]");
    finite(A);finite(b);finite(X);
    double scale=0;
    for(py::ssize_t i=0;i<A.size();++i) scale=std::max(scale,std::abs(A.data()[i]));
    const double tol=64*std::numeric_limits<double>::epsilon()*std::max(1.,scale);
    const int n=int(A.shape(0));
    for(int j=0;j<n;++j) for(int i=0;i<j;++i)
        require(std::abs(A.data()[i+std::size_t(j)*n]-A.data()[j+std::size_t(i)*n])<=tol,
                "two_stage requires symmetric A");
}
struct Context {
    int previous=omp_get_max_threads(),dynamic=omp_get_dynamic();
    Context(int p,int tile,int ib) {
        omp_set_dynamic(0);omp_set_num_threads(p);
        if(plasma_init()!=PlasmaSuccess) {
            omp_set_num_threads(previous);omp_set_dynamic(dynamic);
            throw std::runtime_error("PLASMA initialization failed");
        }
        plasma_set(PlasmaTuning,PlasmaDisabled);
        plasma_set(PlasmaNb,tile);plasma_set(PlasmaIb,ib);
        plasma_set(PlasmaNumPanelThreads,1);
    }
    ~Context() { plasma_finalize();omp_set_num_threads(previous);omp_set_dynamic(dynamic); }
};
struct Plan {
    std::vector<double> diagonal,off,rhs,features;
    int active_workers=0;
    Plan(const double* A,const double* b,const double* X,int n,int c,int s,
         int p,int tile,int ib):diagonal(n),off(n),rhs(std::size_t(n)*c),
         features(std::size_t(s)*n) {
        const int nr=c+s;
        // Rows b^T and X share the SAME right application Q1 Q2.
        std::vector<double> rows(std::size_t(nr)*n),out(rows.size());
        for(int j=0;j<n;++j) {
            for(int k=0;k<c;++k) rows[k+std::size_t(j)*nr]=b[j+std::size_t(k)*n];
            for(int i=0;i<s;++i) rows[c+i+std::size_t(j)*nr]=X[std::size_t(i)*n+j];
        }
        int status;
        {
            // PLASMA uses process-level context maps. Serialize this optional
            // backend; release Python's GIL BEFORE acquiring the mutex.
            std::lock_guard<std::mutex> lock(context_mutex);
            Context context(p,tile,ib);
            status=fed2rc_two_stage_reduce(A,n,rows.data(),nr,tile,ib,
                                           diagonal.data(),off.data(),out.data(),&active_workers);
        }
        if(status) throw std::runtime_error("PLASMA reduction failed: "+std::to_string(status));
        for(int j=0;j<n;++j) {
            for(int k=0;k<c;++k) rhs[j+std::size_t(k)*n]=out[k+std::size_t(j)*nr];
            for(int i=0;i<s;++i) features[i+std::size_t(j)*s]=out[c+i+std::size_t(j)*nr];
        }
    }
};
static py::tuple plan(FArray<double> A,FArray<double> b,CArray<double> X,
                       int threads,int bandwidth,int inner_block) {
    validate(A,b,X,threads,bandwidth,inner_block);
    const int n=int(A.shape(0)),c=int(b.shape(1)),s=int(X.shape(0));
    py::array_t<double> d(n),e(n-1),rhs({n,c},{int(sizeof(double)),int(sizeof(double))*n}),
        h({s,n},{int(sizeof(double)),int(sizeof(double))*s});
    int active=0;
    {
        py::gil_scoped_release release;
        Plan q(A.data(),b.data(),X.data(),n,c,s,threads,bandwidth,inner_block);
        std::copy_n(q.diagonal.data(),n,d.mutable_data());
        std::copy_n(q.off.data(),n-1,e.mutable_data());
        std::copy(q.rhs.begin(),q.rhs.end(),rhs.mutable_data());
        std::copy(q.features.begin(),q.features.end(),h.mutable_data());
        active=q.active_workers;
    }
    return py::make_tuple(d,e,rhs,h,active);
}

// Un buffer per worker, riutilizzato per tutti i lambda assegnati.
// Non servono SPIKE, autovalori o una copia dei pesi per ogni candidato.
struct Workspace {
    std::vector<double> diagonal,off,weights,scores;
    Workspace(int n,int c):diagonal(n),off(n),weights(std::size_t(n)*c),scores(std::size_t(64)*c) {}
};
static void solve_and_score(const Plan& q,const std::int64_t* y,const double* lambdas,
                            int count,int n,int c,int samples,int threads,
                            double* accuracies,int* status) {
    std::vector<Workspace> work;
    for(int i=0;i<threads;++i) work.emplace_back(n,c);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int l=0;l<count;++l) {
        auto& w=work[omp_get_thread_num()];
        int info=0;
        for(int j=0;j<n;++j) {
            w.diagonal[j]=q.diagonal[j]+lambdas[l];
            if(!std::isfinite(w.diagonal[j])) info=-1000;
        }
        std::copy_n(q.off.data(),n-1,w.off.data());
        std::copy(q.rhs.begin(),q.rhs.end(),w.weights.begin());
        if(!info) info=LAPACKE_dptsv_work(LAPACK_COL_MAJOR,n,c,w.diagonal.data(),
                                         w.off.data(),w.weights.data(),n);
        for(int k=0;k<c && !info;++k) {
            double norm=0;
            for(int j=0;j<n;++j) norm+=w.weights[j+std::size_t(k)*n]*w.weights[j+std::size_t(k)*n];
            norm=std::sqrt(norm)+1e-10;
            if(!std::isfinite(norm)) { info=-1000; break; }
            for(int j=0;j<n;++j) w.weights[j+std::size_t(k)*n]/=norm;
        }
        std::int64_t correct=0;
        for(int first=0;first<samples && !info;first+=64) {
            const int rows=std::min(64,samples-first);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,rows,c,n,1,
                        q.features.data()+first,samples,w.weights.data(),n,0,w.scores.data(),64);
            for(int i=0;i<rows;++i) {
                int best=0;
                for(int k=0;k<c;++k) {
                    if(!std::isfinite(w.scores[i+64*k])) info=-1000;
                    if(w.scores[i+64*k]>w.scores[i+64*best]) best=k;
                }
                correct+=best==y[first+i];
            }
        }
        status[l]=info;
        accuracies[l]=info?std::numeric_limits<double>::quiet_NaN():double(correct)/samples;
    }
}
static py::tuple scores(FArray<double> A,FArray<double> b,CArray<double> X,CArray<std::int64_t> y,
                         CArray<double> lambdas,int threads) {
    validate(A,b,X,threads,64,16);
    require(y.ndim()==1 && y.size()==X.shape(0),"invalid y shape");finite(y);
    require(lambdas.ndim()==1 && lambdas.size()>0 && lambdas.size()<=INT_MAX,"invalid lambdas");
    finite(lambdas);
    for(auto i=0;i<lambdas.size();++i) require(lambdas.data()[i]>0,"lambdas must be positive");
    for(auto i=0;i<y.size();++i) require(y.data()[i]>=0 && y.data()[i]<b.shape(1),"invalid class index");
    const int n=int(A.shape(0)),c=int(b.shape(1)),s=int(X.shape(0)),l=int(lambdas.size());
    py::array_t<double> result(l);py::array_t<int> status(l);
    {
        py::gil_scoped_release release;
        Plan q(A.data(),b.data(),X.data(),n,c,s,threads,64,16);
        solve_and_score(q,y.data(),lambdas.data(),l,n,c,s,std::min(threads,l),
                        result.mutable_data(),status.mutable_data());
    }
    return py::make_tuple(result,status);
}
PYBIND11_MODULE(_fed2rc_twostage,m) {
    m.doc()="Optional PLASMA two-stage symmetric reduction; immutable input buffers required.";
    m.def("_plan",&plan,py::arg("A").noconvert(),py::arg("b").noconvert(),
          py::arg("X").noconvert(),py::arg("threads")=1,py::arg("bandwidth")=64,
          py::arg("inner_block")=16);
    m.def("lambda_scores",&scores,py::arg("A").noconvert(),py::arg("b").noconvert(),
          py::arg("X").noconvert(),py::arg("y").noconvert(),py::arg("lambdas").noconvert(),
          py::arg("threads")=1);
    m.attr("plasma_revision")="548e0bdc5a046a728770bccf14503c37732c80fc";
}
