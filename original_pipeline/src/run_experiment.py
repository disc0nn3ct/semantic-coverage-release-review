"""Staged execution; test scoring is blocked until the development selection is locked."""
from __future__ import annotations
import argparse,hashlib,json,pathlib
import numpy as np
from experiment_core import *

CONFIGS=['g3_mx','g3_nomic','gpt_mx','gpt_nomic']
BASELINES=['raw_mx','raw_nomic','codebert','graphcodebert','tfidf','bm25']
SCORES=ROOT/'results/scores'

def verify_seal():
    seal=load_json(ROOT/'protocol/EXECUTION_SEAL.json')
    for relative,expected in seal['files'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen file changed: {relative}')
    return seal

def guard_test(split):
    if split=='test' and not (ROOT/'results/DEVELOPMENT_SELECTION_LOCK.json').exists():
        raise RuntimeError('Test evaluation is locked until development selection has been saved')

def score(corpus,split,method,seed=1701):
    guard_test(split)
    if split=='test' and method in CONFIGS:
        locked=load_json(ROOT/'results/DEVELOPMENT_SELECTION_LOCK.json')
        if method!=locked['chosen_configuration'] or seed not in locked['test_seeds']:
            raise RuntimeError('Scoring of unselected test configurations/seeds is forbidden')
    cases,docs,files,mapping=items(corpus,split)
    if method in ('tfidf','bm25'):
        _,devdocs,_,_=items(corpus,'dev');encoder=Lexical(list(devdocs.values()),method)
        rows=score_cases(cases,files,mapping,lambda fid,did:encoder.coverage(files[fid]['text'],docs[did]))
        metadata={'idf_fit_on':'development descriptions only','method':method,'vocabulary_size':len(encoder.idf)}
    else:
        failed=set();filetexts={fid:f['text'] for fid,f in files.items()}
        if method in CONFIGS:
            llm,embedding=method.split('_');summaries=load_json(ROOT/'results/summaries'/f'{llm}_{split}_{seed}.json')['files']
            if set(summaries)!=set(files):raise RuntimeError('Summary file coverage mismatch')
            failed={fid for fid,s in summaries.items() if not s['valid']}
            filetexts={fid:s['text'] for fid,s in summaries.items() if s['valid']}
        else:embedding=method[4:] if method.startswith('raw_') else method
        if embedding in ('codebert','graphcodebert'):
            from frozen_code_encoders import FrozenEncoder
            encoder=FrozenEncoder(embedding)
            doc_ids=sorted(docs);doc_vectors=encoder.encode([docs[x] for x in doc_ids]).astype(np.float64)
            doc_meta=encoder.last_metadata
            file_ids=sorted(filetexts);file_vectors=encoder.encode([filetexts[x] for x in file_ids]).astype(np.float64)
            metadata={'model':encoder.provenance(),'documents':doc_meta,'files':encoder.last_metadata}
        else:
            from lm_embeddings import LMEmbedding
            encoder=LMEmbedding(embedding)
            doc_ids=sorted(docs);doc_vectors=encoder.encode([docs[x] for x in doc_ids],'document')
            doc_meta=encoder.last_metadata
            file_ids=sorted(filetexts);file_vectors=encoder.encode([filetexts[x] for x in file_ids],'query')
            metadata={'tokenizer':encoder.tokenizer.provenance(),'documents':doc_meta,'files':encoder.last_metadata}
        d={k:v for k,v in zip(doc_ids,doc_vectors)};f={k:v for k,v in zip(file_ids,file_vectors)}
        def similarity(fid,did):
            den=np.linalg.norm(f[fid])*np.linalg.norm(d[did])
            cos=float(np.dot(f[fid],d[did])/den) if den else 0.
            return (1+float(np.clip(cos,-1,1)))/2
        rows=score_cases(cases,files,mapping,similarity,failed)
    result={'created_utc':utc(),'method':method,'split':split,'seed':seed if method in CONFIGS else None,
            'metadata':metadata,'rows':rows}
    path=SCORES/f'{method}_{split}_{seed if method in CONFIGS else "deterministic"}.json'
    save_json(path,result);print(path,flush=True)
    return result

def selection():
    destination=ROOT/'results/DEVELOPMENT_SELECTION_LOCK.json'
    if destination.exists():raise RuntimeError('Selection already locked; will not silently select again')
    configs={};thresholds={}
    for method in CONFIGS+BASELINES:
        path=SCORES/f'{method}_dev_{1701 if method in CONFIGS else "deterministic"}.json'
        source=load_json(path);rows=source['rows'];best,sweep=calibrate(rows)
        thresholds[method]={'threshold':best['threshold'],'fit_metrics':best,
                  'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        save_json(ROOT/'results/calibration'/f'{method}_all_dev.json',sweep)
        if method not in CONFIGS:continue
        oof=[];folds=[];burden=[]
        for cluster in sorted({r['cluster_id'] for r in rows}):
            fit=[r for r in rows if r['cluster_id']!=cluster];held=[r for r in rows if r['cluster_id']==cluster]
            chosen,fold_sweep=calibrate(fit);tau=chosen['threshold'];counts=confusion(held,tau)
            folds.append({'cluster_id':cluster,'threshold':tau,'counts':counts,'sweep':fold_sweep})
            oof.append(counts);burden.extend(sum(f['score']>=tau for f in r['files']) for r in held)
        metrics=classification({k:sum(x[k] for x in oof) for k in ('tp','fp','tn','fn')})
        configs[method]={'oof_metrics':metrics,'oof_mean_flagged_files':sum(burden)/len(burden),'folds':folds}
    chosen=max(CONFIGS,key=lambda c:(configs[c]['oof_metrics']['f1'],configs[c]['oof_metrics']['specificity'],
                     -configs[c]['oof_mean_flagged_files'],-CONFIGS.index(c)))
    result={'locked_utc':utc(),'chosen_configuration':chosen,'primary_comparator':'raw_'+chosen.split('_')[1],
        'configuration_order':CONFIGS,'configuration_selection':configs,'thresholds':thresholds,
        'test_seeds':[1702,1703,1704],'test_results_inspected_before_lock':False,
        'execution_seal_sha256':hashlib.sha256((ROOT/'protocol/EXECUTION_SEAL.json').read_bytes()).hexdigest()}
    save_json(destination,result);print('LOCKED',chosen,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','score','select'])
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--method')
    p.add_argument('--model',choices=['g3','gpt']);p.add_argument('--identifier');p.add_argument('--seed',type=int,default=1701)
    a=p.parse_args();verify_seal();corpus=load_json(ROOT/'data/corpus.json')
    if a.stage=='generate':
        guard_test(a.split)
        if a.split=='test':
            lock=load_json(ROOT/'results/DEVELOPMENT_SELECTION_LOCK.json')
            if a.model!=lock['chosen_configuration'].split('_')[0] or a.seed not in lock['test_seeds']:
                raise RuntimeError('Test model/seed not prespecified')
        generate(corpus,a.split,a.model,a.identifier,a.seed)
    elif a.stage=='score':
        if a.method not in CONFIGS+BASELINES:raise ValueError(a.method)
        score(corpus,a.split,a.method,a.seed)
    else:selection()
if __name__=='__main__':main()
