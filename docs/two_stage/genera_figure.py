"""Generate diagrams from the scheduler's task geometry and measured data.

The Gantt diagram is a UNIT-COST SIMULATION, not a measured execution trace.
Run only after misura_soglia.py completes; no production source is changed.
"""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def scheduler(n=13,nb=4,p=3):
    tasks=[];begin=0
    for i in range(n-1):
        for k in range(3):
            for s in range(begin,i+1):
                t=3*(i-s)+k;first=(t//2)*nb+s+1;last=min(first+nb-1,n-1)
                kind=1 if t==0 else 3-t%2
                done=last>=n-1 or(kind==2 and last>=n-2)
                tasks.append(dict(s=s,t=t,a=first,z=last,kind=kind,owner=(first//nb)%p,done=done))
                if done:begin+=1
    keys={(x['s'],x['t']):x for x in tasks}
    final={x['s']:x['t'] for x in tasks if x['done']}
    for x in tasks:
        s,t=x['s'],x['t'];deps=[]
        if t: deps.append((s,t-1))
        if s:
            source=min(t+2,final[s-1])
            assert source<=t+2<=final[s-1]+2
            deps.append((s-1,source))
        assert all(d in keys for d in deps)
        x['deps']=deps
    return tasks


def main():
    tasks=scheduler();keys={(x['s'],x['t']):x for x in tasks}
    dag=[r'\begin{tikzpicture}[x=2.35cm,y=.76cm,>=Latex]']
    for t in range(5):dag.append(rf'\node[font=\small,text=Ink] at ({t},.8) {{$t={t}$}};')
    for s in range(12):dag.append(rf'\node[font=\small,anchor=east,text=Ink] at (-.65,{-s}) {{$s={s}$}};')
    for x in tasks:
        s,t=x['s'],x['t'];style='draw=Coral,line width=1.3pt' if x['done'] else 'draw=Ink!30'
        dag.append(rf'\node[rounded corners=3pt,fill=P{x["owner"]}!19,{style},minimum width=1.65cm,minimum height=.49cm,font=\small] (k{s}x{t}) at ({t},{-s}) {{$K_{{{s},{t}}}$ / P{x["owner"]}}};')
    for x in tasks:
        s,t=x['s'],x['t']
        for a,b in x['deps']:
            color='Ink!45' if a==s else 'Purple!70'
            opt='dashed' if a!=s else 'solid'
            if (a,b,s,t)==(0,2,1,0):color='Coral';opt='very thick'
            dag.append(rf'\draw[->,{color},{opt},shorten >=1pt,shorten <=1pt] (k{a}x{b}) -- (k{s}x{t});')
    dag.append(r'\end{tikzpicture}')
    (HERE/'fig_dag.tex').write_text('\n'.join(dag))
    pending={p:[x for x in tasks if x['owner']==p] for p in range(3)}
    done=set();schedule=[]
    while any(pending.values()):
        running=[(p,q[0]) for p,q in pending.items() if q and set(map(tuple,q[0]['deps']))<=done]
        assert running,'Deadlock in pedagogical scheduler simulation'
        schedule.append(running)
        for p,x in running:
            pending[p].pop(0);done.add((x['s'],x['t']))
    # Split the long timeline into two readable panels.
    gantt=[r'\begin{tikzpicture}[x=.64cm,y=.82cm,>=Latex]']
    for panel,start in enumerate(range(0,len(schedule),20)):
        offset=-panel*4.4
        for p in range(3):
            y=offset-p
            gantt.append(rf'\node[anchor=east,text=P{p},font=\bfseries\small] at (-.4,{y+.35}) {{P{p}}};')
            for step in range(start,min(start+20,len(schedule))):
                x=step-start;running=dict(schedule[step]);task=running.get(p)
                fill=f'P{p}!35' if task else 'Ink!5'
                gantt.append(rf'\filldraw[fill={fill},draw=white] ({x},{y}) rectangle ++(1,.7);')
                if task:gantt.append(rf'\node[font=\scriptsize] at ({x+.5},{y+.35}) {{{task["s"]}.{task["t"]}}};')
        for step in range(start,min(start+20,len(schedule))):
            gantt.append(rf'\node[font=\scriptsize,anchor=south] at ({step-start+.5},{offset+1}) {{{step}}};')
    gantt.append(r'\end{tikzpicture}')
    (HERE/'fig_timeline.tex').write_text('\n'.join(gantt))
    (HERE/'dati/task_esempio.json').write_text(json.dumps(dict(d=13,nb=4,p=3,tasks=tasks,
        model='unit-cost, worker order from source; not a measured trace',steps=len(schedule)),indent=2))
    Q=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
    T=np.diag([4.,5.,6.,7.])+np.diag(np.ones(3),1)+np.diag(np.ones(3),-1)
    A=Q@T@Q.T;b=np.arange(1.,5.);X=np.array([[2.,0.,1.,3.]])
    U=np.linalg.solve(T+np.eye(4),Q.T@b);W=np.linalg.solve(A+np.eye(4),b)
    assert np.allclose(W,Q@U) and np.allclose(X@W,(X@Q)@U)
    example=dict(A=A.tolist(),T=T.tolist(),Q=Q.tolist(),b=b.tolist(),g=(Q.T@b).tolist(),
                 H=(X@Q).tolist(),U=U.tolist(),W=W.tolist(),score=float((X@W)[0]))
    (HERE/'dati/esempio_numerico.json').write_text(json.dumps(example,indent=2))
    def matrix(a):
        a=np.atleast_2d(a)
        return r'\begin{bmatrix}'+r'\\'.join(' & '.join(f'{v:g}' for v in row) for row in a)+r'\end{bmatrix}'
    (HERE/'dati/esempio_numerico.tex').write_text(
        '\\newcommand{\\ExampleA}{'+matrix(A)+'}\n'+
        '\\newcommand{\\ExampleG}{'+matrix((Q.T@b)[:,None])+'}\n'+
        '\\newcommand{\\ExampleH}{'+matrix(X@Q)+'}\n'+
        f'\\newcommand{{\\ExampleScore}}{{{example["score"]:.8f}}}\n')
    data=json.loads((HERE/'dati/misure.json').read_text());assert data['complete']
    groups={}
    for r in data['records']:groups.setdefault((r['d'],r['threads']),[]).append(r)
    threshold_rows=[];hundred_rows=[];summary=[]
    for (d,p),items in sorted(groups.items()):
        ordered=sorted((r for r in items if r['L']<=32),key=lambda r:r['L'])
        first=next((r['L'] for r in ordered if r['speedup']>1),None)
        robust=next((r['L'] for i,r in enumerate(ordered[:-2]) if all(q['speedup']>=1.1 for q in ordered[i:i+3])),None)
        hundred=next(r for r in items if r['L']==100)
        threshold_rows.append(f"{d} & {p} & {first or '--'} & {robust or '--'} \\\\")
        hundred_rows.append(f"{d} & {p} & {hundred['lu_ms']:.2f} & {hundred['two_ms']:.2f} & {hundred['speedup']:.2f} \\\\")
        summary.append(dict(d=d,p=p,first_observed=first,three_consecutive_10percent=robust,at_100=hundred))
    (HERE/'dati/soglie_tabella.tex').write_text('\\newcommand{\\ThresholdRows}{\n'+'\n'.join(threshold_rows)+'\n}\n')
    (HERE/'dati/cento_tabella.tex').write_text('\\newcommand{\\HundredRows}{\n'+'\n'.join(hundred_rows)+'\n}\n')
    (HERE/'dati/soglie_riassunto.json').write_text(json.dumps(dict(records=summary),indent=2))
    print('Generated exact-geometry DAG:',len(tasks),'tasks;',len(schedule),'unit-cost steps.')
    print('Crossover summaries:',[(r['d'],r['p'],r['first_observed'],r['three_consecutive_10percent']) for r in summary])


if __name__=='__main__': main()
