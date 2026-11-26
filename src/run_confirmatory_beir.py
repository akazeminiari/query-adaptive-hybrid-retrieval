from pathlib import Path
import re, time
import numpy as np
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from beir import util
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'router'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ['query_length','unique_terms','bm25_top1','bm25_top2','bm25_margin','bm25_top5_mean']
EXTRA = ['top1_query_coverage','top5_query_coverage','top1_title_coverage','bm25_score_drop','top5_document_diversity']
FEATURES = BASE + EXTRA
DATASETS = ('trec-covid',)
MARGIN_THRESHOLD = 5.12617462
B_RANDOM = 50000
B_BOOT = 20000

def tok(s):
    return re.findall(r'\b\w+\b', (s or '').lower())

def model_factory():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))

def extra_features(qset, top, scores, token_sets, title_sets):
    coverage = [len(qset & token_sets[i])/len(qset) if qset else 0. for i in top]
    similarities = []
    for i in range(5):
        for j in range(i+1, 5):
            a, b = token_sets[top[i]], token_sets[top[j]]
            similarities.append(len(a & b)/len(a | b) if a | b else 0.)
    return [coverage[0], float(np.mean(coverage)), len(qset & title_sets[top[0]])/len(qset) if qset else 0., float(scores[top[0]]-scores[top[4]]), 1-float(np.mean(similarities))]

def rrf(a,b,k=10):
    s={}
    for ranklist in (a,b):
        for rank,doc in enumerate(ranklist,1):
            s[doc]=s.get(doc,0.)+1/(k+rank)
    return sorted(s,key=lambda d:(-s[d],d))[:5]

def ndcg(rank,rel):
    actual=[float(rel.get(d,0)) for d in rank[:5]]
    ideal=sorted((float(x) for x in rel.values()),reverse=True)[:5]
    dcg=sum((2**g-1)/np.log2(i+2) for i,g in enumerate(actual))
    idcg=sum((2**g-1)/np.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0.

def evaluate(name, model, threshold, encoder):
    data=ROOT/'datasets'/name
    if not (data/'corpus.jsonl').exists():
        url=f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'
        print('Downloading',url,flush=True)
        util.download_and_unzip(url,str(ROOT/'datasets'))
    docs,queries,qrels=GenericDataLoader(data_folder=str(data)).load(split='test')
    ids=list(docs)
    texts=[(docs[d].get('title','')+' '+docs[d].get('text','')).strip() for d in ids]
    print(f'[{name}] corpus={len(ids)} test queries={len(qrels)}',flush=True)
    tokens=[tok(t) for t in texts]
    token_sets=[set(t) for t in tokens]
    title_sets=[set(tok(docs[d].get('title',''))) for d in ids]
    bm25=BM25Okapi(tokens)
    print(f'[{name}] encoding corpus on CPU; excluded from query latency',flush=True)
    emb=encoder.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
    encoder.encode(['warmup'],normalize_embeddings=True,convert_to_numpy=True)
    rows=[]
    for n,qid in enumerate(sorted(qrels),1):
        query=queries[qid]; qt=tok(query)
        start=time.perf_counter()
        scores=bm25.get_scores(qt)
        top=np.argsort(-scores)[:5]
        b_rank=[ids[j] for j in top]
        t_bm=time.perf_counter()
        vals=[len(qt),len(set(qt)),float(scores[top[0]]),float(scores[top[1]]),float(scores[top[0]]-scores[top[1]]),float(np.mean(scores[top]))]
        vals+=extra_features(set(qt),top,scores,token_sets,title_sets)
        prob=float(model.predict_proba(np.asarray(vals).reshape(1,-1))[0,1])
        run_dense=int(prob>=threshold)
        margin_dense=int(vals[4]<=MARGIN_THRESHOLD)
        t_router=time.perf_counter()
        qemb=encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0]
        d_scores=emb@qemb
        dtop=np.argsort(-d_scores)[:5]
        h_rank=rrf(b_rank,[ids[j] for j in dtop],k=10)
        t_dense=time.perf_counter()
        b=ndcg(b_rank,qrels[qid]); h=ndcg(h_rank,qrels[qid])
        rows.append(dict(query_id=qid,probability=prob,run_dense=run_dense,bm25_margin=vals[4],margin_run_dense=margin_dense,
                         bm25_ndcg=b,hybrid_ndcg=h,router_ndcg=h if run_dense else b,
                         frozen_margin_ndcg=h if margin_dense else b,
                         bm25_ms=(t_bm-start)*1000,router_ms=(t_router-t_bm)*1000,dense_rrf_ms=(t_dense-t_router)*1000))
        if n%25==0 or n==len(qrels):
            print(f'[{name}] {n}/{len(qrels)}',flush=True)
    res=pd.DataFrame(rows)
    path=OUT/f'{name}_confirmatory_queries.csv'
    res.to_csv(path,index=False)
    n=len(res); k=int(res.run_dense.sum())
    b=res.bm25_ndcg.to_numpy(); h=res.hybrid_ndcg.to_numpy()
    gain=h-b; router=res.router_ndcg.to_numpy()
    rng=np.random.default_rng(20261103)
    rand_scores=np.empty(B_RANDOM)
    for i in range(B_RANDOM):
        chosen=rng.choice(n,size=k,replace=False)
        rand_scores[i]=b.mean()+gain[chosen].sum()/n
    p=(1+int(np.count_nonzero(rand_scores>=router.mean()-1e-14)))/(B_RANDOM+1)
    rng_boot=np.random.default_rng(20261104)
    def ci(other):
        diff=router-other
        samples=np.empty(B_BOOT)
        for j in range(B_BOOT):
            samples[j]=diff[rng_boot.integers(0,n,size=n)].mean()
        return tuple(np.quantile(samples,[.025,.975]))
    ci_bm=ci(b); ci_h=ci(h); ci_margin=ci(res.frozen_margin_ndcg.to_numpy())
    hybrid_est=(res.bm25_ms+res.dense_rrf_ms).mean()
    router_est=(res.bm25_ms+res.router_ms+res.run_dense*res.dense_rrf_ms).mean()
    result=dict(dataset=name,queries=n,router_dense_calls=k,router_dense_fraction=k/n,
                bm25_ndcg=b.mean(),hybrid_ndcg=h.mean(),router_ndcg=router.mean(),
                frozen_margin_ndcg=res.frozen_margin_ndcg.mean(),margin_dense_calls=int(res.margin_run_dense.sum()),
                fixed_call_random_mean_ndcg=rand_scores.mean(),router_minus_random=router.mean()-rand_scores.mean(),
                primary_one_sided_p=p,random_95pct_sim_low=np.quantile(rand_scores,.025),random_95pct_sim_high=np.quantile(rand_scores,.975),
                router_minus_bm25_ci_low=ci_bm[0],router_minus_bm25_ci_high=ci_bm[1],
                router_minus_hybrid_ci_low=ci_h[0],router_minus_hybrid_ci_high=ci_h[1],
                router_minus_frozen_margin_ci_low=ci_margin[0],router_minus_frozen_margin_ci_high=ci_margin[1],
                hybrid_component_est_ms=hybrid_est,router_component_est_ms=router_est)
    print(f'[{name}] Router={router.mean():.6f}; random fixed-call={rand_scores.mean():.6f}; p={p:.6f}; calls={k}/{n}',flush=True)
    print(f'[{name}] component-accounted latency only: Hybrid={hybrid_est:.3f} ms Router={router_est:.3f} ms',flush=True)
    print('Saved:',path,flush=True)
    return result

def main():
    train=pd.read_csv(ROOT/'data'/'scifact'/'router_train_enriched.csv',dtype={'query_id':str})
    train=train[train['split'].eq('train')].reset_index(drop=True)
    if len(train)!=647: raise ValueError(f'Expected 647 SciFact train queries, got {len(train)}')
    X=train[FEATURES].to_numpy(dtype=float)
    y=train.hybrid_helpful.to_numpy(dtype=int)
    oof=np.empty(len(y))
    for tr,ho in StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y):
        m=model_factory();m.fit(X[tr],y[tr]);oof[ho]=m.predict_proba(X[ho])[:,1]
    threshold=float(np.quantile(oof,.5))
    model=model_factory();model.fit(X,y)
    print(f'Frozen SciFact model: n={len(train)}, threshold={threshold:.8f}',flush=True)
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    results=[evaluate(name,model,threshold,encoder) for name in DATASETS]
    p=np.asarray([r['primary_one_sided_p'] for r in results]); order=np.argsort(p)
    adjusted=np.empty(len(results))
    running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(results)-rank)*p[index]);adjusted[index]=min(1.,running)
    for r,adj in zip(results,adjusted):
        r['holm_adjusted_p']=float(adj)
        r['confirmatory_positive_at_0_05']=bool(adj<=.05 and r['router_minus_random']>0)
    summary=OUT/'confirmatory_beir_summary.csv'
    pd.DataFrame(results).to_csv(summary,index=False)
    print('\n=== CONFIRMATORY SUMMARY ===',flush=True)
    print(pd.DataFrame(results)[['dataset','router_ndcg','fixed_call_random_mean_ndcg','router_minus_random','primary_one_sided_p','holm_adjusted_p','confirmatory_positive_at_0_05']].to_string(index=False),flush=True)
    print('Saved:',summary,flush=True)
    print('CAUTION: primary reference matches Dense call COUNT, not measured wall-clock time. Component-accounted timings are NOT direct benchmarks.',flush=True)

if __name__=='__main__':
    main()

