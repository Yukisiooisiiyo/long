import sys,json,time; sys.path.insert(0,'.'); sys.path.insert(0,'bench')
from run import INSTANCES, BEST
from lp import heuristics as H, lagrange as LG
from lp.graph import identity_cgraph
from lp.solver import _cg_path_to_orig
out=open('results/init.jsonl','w')
opt={}
for l in open('results/best.jsonl'):
    r=json.loads(l)
    if r['cfg']=='BEST': opt[r['inst']]=r['len']
for i in ["rail80","rail120","rail160","grid5x5","grid5x6","grid6x6","dense16","dense18","dense22"]:
    g=INSTANCES[i](); cg=identity_cgraph(g)
    res={}
    def t(name,f):
        t0=time.perf_counter(); v=f(); res[name]=(round(v[0],2), round(time.perf_counter()-t0,3)); return v
    t('greedy',lambda:H.greedy(g)); m=t('mst',lambda:H.mst_diameter(g)); t('fcycle',lambda:H.fundamental_cycles(g)); b=t('beam',lambda:H.beam(g,K=32))
    def lag():
        lam,ub,ch,_=LG.subgradient(cg,b[0],100)
        best=(0,[])
        for L,seq,es in LG.forest_paths(cg,ch)[:3]:
            p=_cg_path_to_orig(cg,seq,es); v=H.extend_greedy(g,p)
            if v[0]>best[0]: best=v
        return best
    t('lagr',lag)
    pb=b[1][:-1] if b[1][0]==b[1][-1] else b[1]
    t('beam+ils',lambda:H.ils(g,pb,30,0))
    t('mst+ils',lambda:H.ils(g,m[1],30,0))
    rec={'inst':i,'opt':round(opt[i],2),**{k:v for k,v in res.items()}}
    out.write(json.dumps(rec)+'\n'); print(rec,flush=True)
