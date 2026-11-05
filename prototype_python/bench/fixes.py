import sys,json; sys.path.insert(0,'.'); sys.path.insert(0,'bench')
from run import INSTANCES, FULL, o
from dataclasses import replace
from lp.solver import solve
cfgs={'FULL':FULL,'compress+local':o(FULL,partial_bound='local'),'-compress':replace(FULL,compress=False),
 'FULL+1forest':replace(o(FULL,cycle_1forest=True),lagr_cycle=True),
 '-compress+1forest':replace(o(FULL,cycle_1forest=True),lagr_cycle=True,compress=False)}
out=open('results/fixes.jsonl','w')
for i in ["rail80","rail120","grid5x5","grid5x6","dense16","dense18"]:
    g=INSTANCES[i]()
    for c,cfg in cfgs.items():
        (L,p),st=solve(g,cfg,time_limit=60)
        r={'inst':i,'cfg':c,'len':L,**st}; out.write(json.dumps(r)+'\n'); out.flush()
        print(i,c,round(L,2),st.get('nodes'),round(st['time'],2),'del',st.get('edges_deleted'),'proved',st.get('proved_by_lagr'),'cub',round(st.get('cycle_ub',0),2),flush=True)
