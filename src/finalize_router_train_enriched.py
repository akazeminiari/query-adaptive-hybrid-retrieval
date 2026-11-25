import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

BASE = ['query_length','unique_terms','bm25_top1','bm25_top2','bm25_margin','bm25_top5_mean']
EXTRA = ['top1_query_coverage','top5_query_coverage','top1_title_coverage',
         'bm25_score_drop','top5_document_diversity']
ORIGINAL_FIELDS = ['query_id','split'] + BASE + ['bm25_ndcg','hybrid_ndcg','hybrid_gain','hybrid_helpful']
SEED = 42
EXPECTED_ROWS = 647
EXPECTED_POSITIVES = 134
EXPECTED_THRESHOLD = 0.4994615425
THRESHOLD_ATOL = 1e-6
SCORE_ATOL = 1e-4


def tok(s):
    return re.findall(r'\b\w+\b', (s or '').lower())


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_source(path):
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns is None or not set(ORIGINAL_FIELDS).issubset(columns):
            raise ValueError('Unexpected router_dataset.csv columns: ' + str(columns))
        rows = list(reader)
    if len(rows) != 809:
        raise ValueError(f'Expected 809 source rows (647 train + 162 val); found {len(rows)}')
    split_counts = {split: sum(r['split'] == split for r in rows) for split in ('train','val')}
    if split_counts != {'train':647, 'val':162}:
        raise ValueError(f'Unexpected SciFact source split counts: {split_counts}')
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate query IDs detected')
    train = [r for r in rows if r['split'] == 'train']
    if len(train) != EXPECTED_ROWS:
        raise ValueError('Unexpected SciFact train-row count')
    pos = 0
    for r in train:
        label = int(r['hybrid_helpful'])
        if label not in (0,1):
            raise ValueError('Nonbinary label at query ' + r['query_id'])
        b, h = float(r['bm25_ndcg']), float(r['hybrid_ndcg'])
        if label != int(h > b):
            raise ValueError('hybrid_helpful rule mismatch at query ' + r['query_id'])
        if abs(float(r['hybrid_gain']) - (h-b)) > 1e-9:
            raise ValueError('hybrid_gain mismatch at query ' + r['query_id'])
        for col in BASE:
            if not math.isfinite(float(r[col])):
                raise ValueError(f'Nonfinite base feature {col} at {r["query_id"]}')
        pos += label
    if pos != EXPECTED_POSITIVES:
        raise ValueError(f'Expected {EXPECTED_POSITIVES} positive labels; found {pos}')
    return train, split_counts, pos, columns


def compute_extra(train, data_dir):
    import numpy as np
    from rank_bm25 import BM25Okapi
    from beir.datasets.data_loader import GenericDataLoader

    docs, queries, _ = GenericDataLoader(data_folder=str(data_dir)).load(split='train')
    ids = list(docs)
    texts = [(docs[i].get('title','')+' '+docs[i].get('text','')).strip() for i in ids]
    tokens = [tok(s) for s in texts]
    token_sets = [set(s) for s in tokens]
    title_sets = [set(tok(docs[i].get('title',''))) for i in ids]
    bm25 = BM25Okapi(tokens)
    output = []
    score_errors = {name: [] for name in ['bm25_top1','bm25_top2','bm25_margin','bm25_top5_mean']}

    for idx, record in enumerate(train, 1):
        qid = record['query_id']
        if qid not in queries:
            raise ValueError(f'Train query {qid} is absent from BEIR queries.jsonl')
        qt = tok(queries[qid]); qset = set(qt)
        scores = bm25.get_scores(qt)
        top = np.argsort(-scores)[:5]
        if len(top) != 5:
            raise ValueError('Fewer than 5 BM25 candidates')
        actual_base = {
            'bm25_top1': float(scores[top[0]]),
            'bm25_top2': float(scores[top[1]]),
            'bm25_margin': float(scores[top[0]]-scores[top[1]]),
            'bm25_top5_mean': float(np.mean(scores[top]))
        }
        for name, actual in actual_base.items():
            score_errors[name].append(abs(actual - float(record[name])))
        coverage = [len(qset & token_sets[i])/len(qset) if qset else 0.0 for i in top]
        title_cov = len(qset & title_sets[top[0]])/len(qset) if qset else 0.0
        pairs = [jaccard(token_sets[top[i]], token_sets[top[j]])
                 for i in range(5) for j in range(i+1,5)]
        # Identical definitions and numpy aggregation as router_feature_ablation.py.
        output.append({
            'top1_query_coverage': coverage[0],
            'top5_query_coverage': float(np.mean(coverage)),
            'top1_title_coverage': title_cov,
            'bm25_score_drop': float(scores[top[0]]-scores[top[4]]),
            'top5_document_diversity': float(1.0-np.mean(pairs))
        })
        if idx % 150 == 0 or idx == len(train):
            print(f'SciFact additional features: {idx}/{len(train)}', flush=True)

    max_diffs = {name: max(errors) for name, errors in score_errors.items()}
    if any(error > SCORE_ATOL for error in max_diffs.values()):
        raise ValueError('Source BM25 score fingerprint mismatch: ' + json.dumps(max_diffs) +
                         '. Do not publish; check BEIR corpus/order/preprocessing.')
    return output, max_diffs, len(ids)


def verify_threshold(rows):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features = BASE + EXTRA
    X = np.asarray([[float(r[c]) for c in features] for r in rows], dtype=float)
    y = np.asarray([int(r['hybrid_helpful']) for r in rows], dtype=int)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite values among 11 features')
    for c in ['top1_query_coverage','top5_query_coverage','top1_title_coverage','top5_document_diversity']:
        if any(not (-1e-12 <= float(r[c]) <= 1+1e-12) for r in rows):
            raise ValueError('Coverage/diversity outside [0,1] for ' + c)
    if any(float(r['bm25_score_drop']) < -1e-9 for r in rows):
        raise ValueError('Negative BM25 score-drop')
    oof = np.empty(len(y), dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in folds.split(X,y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=SEED))
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:,1]
    median = float(np.quantile(oof, 0.5))
    if abs(median - EXPECTED_THRESHOLD) > THRESHOLD_ATOL:
        raise ValueError('Frozen OOF median does NOT reproduce: observed=' + str(median) +
                         ', reference=' + str(EXPECTED_THRESHOLD) +
                         '. Keep original artifact; do not retune to target data.')
    return median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],
                        help='Project root (e.g. E:\\rag_windows_starter)')
    parser.add_argument('--overwrite',action='store_true',help='Explicitly replace pre-existing output if it differs')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root/'outputs'/'router'
    base_path = out/'router_dataset.csv'
    data_dir = root/'datasets'/'scifact'
    target_path = out/'router_train_enriched.csv'
    report_path = out/'router_train_enriched_validation.json'
    print('Project root:', root, flush=True)
    if not base_path.exists():
        raise FileNotFoundError('Missing source: ' + str(base_path))
    train, splits, positives, columns = read_source(base_path)
    print(f'Validated source: 647 train, 162 val, {positives} positive train labels',flush=True)
    required = [data_dir/'corpus.jsonl',data_dir/'queries.jsonl',data_dir/'qrels'/'train.tsv']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('BEIR SciFact source files required. Missing:\n  ' + '\n  '.join(missing))
    extra, max_diffs, n_docs = compute_extra(train,data_dir)
    enriched = [dict(record, **new) for record,new in zip(train,extra)]
    threshold = verify_threshold(enriched)
    fieldnames = list(columns)+EXTRA
    report = {
        'status':'verified_against_source_BM25_and_frozen_OOF_threshold',
        'source':str(base_path),
        'output':str(target_path),
        'train_rows':len(enriched),
        'source_splits':splits,
        'positive_labels':positives,
        'feature_count':11,
        'features':BASE+EXTRA,
        'source_corpus_documents':n_docs,
        'bm25_max_absolute_score_differences':max_diffs,
        'frozen_oof_median_recomputed':threshold,
        'frozen_oof_median_reference':EXPECTED_THRESHOLD,
        'source_sha256':sha256(base_path),
        'scifact_corpus_sha256':sha256(required[0]),
        'scifact_queries_sha256':sha256(required[1]),
        'scifact_train_qrels_sha256':sha256(required[2]),
        'python_version':sys.version,
        'platform':platform.platform()
    }
    if target_path.exists() and not args.overwrite:
        # Existing historical artifact is left unchanged and can be examined separately.
        with target_path.open('r',newline='',encoding='utf-8-sig') as fh:
            existing = list(csv.DictReader(fh))
        if len(existing) != len(enriched) or any(
            any(abs(float(existing[i][c])-float(enriched[i][c])) > 1e-10 for c in BASE+EXTRA)
            or existing[i]['query_id'] != enriched[i]['query_id']
            or int(existing[i]['hybrid_helpful']) != int(enriched[i]['hybrid_helpful'])
            for i in range(min(len(existing),len(enriched)))
        ):
            raise FileExistsError('Existing router_train_enriched.csv differs. Preserved unchanged. '
                                  'Examine differences first; only use --overwrite if justified.')
        print('Existing enriched CSV matches independently recomputed features; no overwrite needed.')
    else:
        out.mkdir(parents=True,exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.router_train_enriched_',suffix='.csv',dir=str(out))
        try:
            with os.fdopen(fd,'w',newline='',encoding='utf-8') as fh:
                writer = csv.DictWriter(fh,fieldnames=fieldnames,extrasaction='ignore')
                writer.writeheader()
                writer.writerows(enriched)
            os.replace(tmp,target_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        print('Wrote final verified CSV:',target_path,flush=True)
    report['output_sha256'] = sha256(target_path)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Wrote reproducibility report:',report_path,flush=True)
    print(f'Confirmed frozen threshold = {threshold:.10f}; do not recalibrate.',flush=True)

if __name__=='__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print('VALIDATION STOPPED:',exc,file=sys.stderr)
        sys.exit(2)

