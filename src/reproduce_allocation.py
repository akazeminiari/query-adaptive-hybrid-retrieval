import csv
from pathlib import Path
import numpy as np

SEED=20261109
DRAWS=50000
DATASETS=[('SCIDOCS','scidocs_per_query(1).csv'),('CQADupstack-Android','cqadupstack_android_per_query(1).csv')]
output=[]
for name, filename in DATASETS:
    with open(Path(__file__).parent.parent / filename, encoding='utf-8-sig', newline='') as f:
        data=list(csv.DictReader(f))
    n=len(data)
    bm25=np.array([float(row['bm25_ndcg5']) for row in data]); hybrid=np.array([float(row['hybrid_ndcg5']) for row in data]);
    router=np.array([float(row['router11_ndcg5']) for row in data]); routed=np.array([int(row['router11_dense']) for row in data]);
    gain=hybrid-bm25; k=int(routed.sum())
    assert np.allclose(router,np.where(routed==1,hybrid,bm25),atol=1e-12)
    expectation=bm25.mean()+(k/n)*gain.mean()
    rng=np.random.default_rng(SEED)
    scores=np.empty(DRAWS)
    for i in range(DRAWS):
        indices=rng.choice(n,size=k,replace=False)
        scores[i]=bm25.mean()+gain[indices].sum()/n
    exceed=int(np.count_nonzero(scores >= router.mean()-1e-14))
    p=(1+exceed)/(DRAWS+1)
    print(name, 'N', n, 'K', k, 'BM25',bm25.mean(),'Hybrid',hybrid.mean(),'Router',router.mean(),'Exact Random E',expectation,'Monte Carlo mean',scores.mean(),'exceed',exceed,'p',p)
    output.append(dict(dataset=name,n=n,k=k,bm25_mean=bm25.mean(),hybrid_mean=hybrid.mean(),router_mean=router.mean(),random_exact_expectation=expectation,random_mc_mean=scores.mean(),router_minus_random_exact=router.mean()-expectation,random_mc_p025=np.percentile(scores,2.5),random_mc_p975=np.percentile(scores,97.5),extreme_draws=exceed,one_sided_p=p))
# Holm two-comparison family
order=sorted(range(len(output)),key=lambda i:output[i]['one_sided_p'])
maxadj=0
for rank,i in enumerate(order):
    maxadj=max(maxadj,min(1,(len(output)-rank)*output[i]['one_sided_p']))
    output[i]['holm_adjusted_p']=maxadj
with open(Path(__file__).parent/'SCIDOCS_Android_statistical_audit.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
