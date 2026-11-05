import sys; sys.path.insert(0,'.'); sys.path.insert(0,'bench')
from run import INSTANCES, FULL
from lp.graph import compress, identity_cgraph
from lp.bnb import Search
from lp.solver import initial_solution, Stats
from dataclasses import replace
for iname in ['grid5x6','rail80','dense18']:
    g=INSTANCES[iname]()
    for comp in [True, False]:
        cg=compress(g) if comp else identity_cgraph(g)
        st=Stats()
        best,lag=initial_solution(g,cg,FULL,st)
        lam=lag[0]
        s=Search(cg,FULL.bnb,best[0],best[1],lam)
        ds=s.start_descriptors()
        nv=npart=0
        for d in ds:
            n0=s.nodes; s.run_path(descriptors=[d]); 
            if d[0]=='v': nv+=s.nodes-n0
            else: npart+=s.nodes-n0
        s2=Search(cg,FULL.bnb,s.best,s.best_path,lam); s2.run_cycle()
        print(iname,'compress' if comp else 'plain','cg.n',cg.n,'chains',sum(1 for e in cg.edges if e[5]),'max_partial',round(cg.max_partial,2),'path nodes: vertex-starts',nv,'partial-starts',npart,'| cycle nodes',s2.nodes)
