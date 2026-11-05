import sys,json; sys.path.insert(0,'.'); sys.path.insert(0,'bench')
from run import INSTANCES, FULL, o
from dataclasses import replace
from lp.solver import solve
base=replace(o(FULL,cycle_1forest=True),lagr_cycle=True,compress=False)
out=open('results/depth.jsonl','w')
for i in ["grid5x6","dense16","dense18","rail80"]:
    g=INSTANCES[i]()
    for d in [0,1,3,6,10,99]:
        cfg=o(base,lagr_depth=d)
        (L,p),st=solve(g,cfg,time_limit=60)
        r={'inst':i,'depth':d,'len':L,**st}; out.write(json.dumps(r)+'\n'); out.flush()
        print(i,d,round(L,2),st.get('nodes'),round(st['time'],2),flush=True)
