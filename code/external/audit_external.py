"""Independent offline audit; no imports from the experimental metric code.

Usage: python audit_external.py [external_validation_root]
All writes stay beside this script. Source data and model calls are read-only.
"""
from pathlib import Path
from collections import Counter, defaultdict
import ast, csv, hashlib, json, math, statistics, sys, zipfile
import numpy as np

OUT = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv)>1 else OUT.parents[1]/'external_gold_validation_20260930'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value): (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
checks=[]
def check(name, yes, detail=None):
    checks.append({'check':name,'passed':bool(yes),**({'detail':detail} if detail is not None else {})})
def csvsave(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(y,p):
    tp=sum(a==1 and b==1 for a,b in zip(y,p)); fp=sum(a==0 and b==1 for a,b in zip(y,p))
    tn=sum(a==0 and b==0 for a,b in zip(y,p)); fn=sum(a==1 and b==0 for a,b in zip(y,p))
    abp=sum(a==1 and b==-1 for a,b in zip(y,p)); abn=sum(a==0 and b==-1 for a,b in zip(y,p))
    n=len(y);pos=sum(y);neg=n-pos;bn=tp+fp+tn+fn
    agree=(tp+tn)/bn if bn else None
    pe=((tp+fn)*(tp+fp)+(tn+fp)*(tn+fn))/(bn*bn) if bn else None
    return dict(N=n,Positive=pos,Negative=neg,BinaryN=bn,TP=tp,FP=fp,TN=tn,FN=fn,AbstPositive=abp,AbstNegative=abn,
       Precision=tp/(tp+fp) if tp+fp else 0,Recall=tp/pos,Specificity=tn/neg,
       F1=2*tp/(2*tp+fp+fn+abp) if 2*tp+fp+fn+abp else 0,
       Accuracy=(tp+tn)/n,BalancedAccuracy=(tp/pos+tn/neg)/2,
       Kappa=(agree-pe)/(1-pe) if bn and pe!=1 else None)
def auc_ap(y,s):
    positive=[v for a,v in zip(y,s) if a==1];negative=[v for a,v in zip(y,s) if a==0]
    auc=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
    groups=defaultdict(list)
    for a,b in zip(y,s):groups[b].append(a)
    seen=0;tp=0;ap=0
    for v in sorted(groups,reverse=True):
        g=groups[v];np_=sum(g);seen+=len(g);tp+=np_;ap+=np_/len(positive)*tp/seen
    return auc,ap
def parse_raw(raw):
    if raw.get('status')!='ok':return 'parse_failure',None,'transport_failure'
    choice=raw['response']['choices'][0]
    if choice.get('finish_reason')!='stop':return 'parse_failure',None,'non-stop'
    txt=choice['message']['content'].strip()
    dec=json.JSONDecoder(); objects=[]
    for i,ch in enumerate(txt):
        if ch!='{':continue
        try:obj,_=dec.raw_decode(txt[i:])
        except json.JSONDecodeError:continue
        if isinstance(obj,dict) and 'classification' in obj:objects.append(obj)
    if len(objects)!=1:return 'parse_failure',None,'object_count'
    obj=objects[0]
    if obj.get('classification') not in {'consistent','inconsistent','uncertain','not_assessable'} or not isinstance(obj.get('evidence'),str) or not obj['evidence'].strip():
        return 'parse_failure',obj,'schema'
    return obj['classification'],obj,None
def clean_doc(doc):
    doc=doc.strip()
    if doc.startswith('/**') and doc.endswith('*/'):
        doc='\n'.join(line.lstrip().removeprefix('*').removeprefix(' ') for line in doc[3:-2].splitlines())
    if doc.startswith('"""') and doc.endswith('"""'):doc=doc[3:-3]
    return doc.strip()

def main():
    protocol=read(SRC/'protocol/PROTOCOL.json');lock=read(SRC/'protocol/EXECUTION_LOCK.json')
    runnerlock=read(SRC/'protocol/RUNNER_IMPLEMENTATION_LOCK.json')
    manifest=read(SRC/'data/reference_manifest.json');cases=manifest['cases'];caseby={c['case_id']:c for c in cases}
    inputs=read(SRC/'data/blinded_inputs.json');inputby={c['case_id']:c for c in inputs}
    oldmetrics=read(SRC/'results/EXTERNAL_RESULTS.json')
    for rel,h in lock['files'].items():check('frozen '+rel,sha(SRC/rel)==h)
    check('runner source hash',sha(SRC/'run_external.py')==runnerlock['implementation_sha256'])
    for rec in read(SRC/'data/source_sha256_manifest.json'):
        check('upstream source hash '+rec['path'],sha(SRC/rec['path'])==rec['sha256'])
    check('case uniqueness',len(cases)==len(caseby)==194)
    check('inputs same ids',set(caseby)==set(inputby))
    upstream_doc={}
    for project in ['pandas','requests']:
        folder=SRC/'sources/DocPrism/datasets/ablation_dataset'
        for obj in read(folder/(project+'_labeled.json')):upstream_doc[f'DocPrism/{project}/{obj["id"]}']=obj
    for c in cases:
        inp=inputby[c['case_id']]
        check('blinded fields '+c['case_id'],set(inp)=={'case_id','code_id','language','code','document','code_sha256','document_sha256'})
        check('input match '+c['case_id'],all(c[k]==inp[k] for k in inp))
        if c['cohort']=='DocPrism':
            u=upstream_doc[c['source_id']]
            check('reference polarity '+c['case_id'],c['reference_label']==int(u['label']=='inconsistent'))
            check('upstream content '+c['case_id'],c['code']==u['code'].strip() and c['document']==clean_doc(u['doc']))
        else:
            u=read(SRC/c['source_member'])[0];s=u['signature']
            signature=''.join(s.get('modifier',[]))
            signature+=('<'+', '.join(s['generics'])+'> ') if s.get('generics') else ''
            signature+=s.get('returns','')+' '+s['name']+'('+', '.join(s.get('params',[]))+')'
            signature+=(' throws '+', '.join(s['exceptions'])) if s.get('exceptions') else ''
            code=(signature+' '+u['code']).strip()
            label=(SRC/c['source_member']).with_name('inconsistency.txt').read_text().strip()
            check('reference polarity '+c['case_id'],c['reference_label']==int(label=='True'))
            check('upstream content '+c['case_id'],c['code']==code and c['document']==clean_doc(u['doc']))

    votes={};rawrows=[];bad_schema=[];raw_details={};request_times=[];finish=Counter()
    csvruns=list(csv.DictReader((SRC/'results/assessor_case_runs.csv').open(encoding='utf-8')))
    csvlookup={(r['model'],r['case_id'],int(r['seed'])):r for r in csvruns}
    oldvotes={(r['model'],r['case_id']):r for r in read(SRC/'results/consensus_predictions.json')}
    for model in protocol['assessor_models']:
        for c in cases:
            cats=[]
            for seed in protocol['assessor_seeds']:
                path=SRC/'results/assessors'/model/c['case_id']/f'raw_seed_{seed}.json'
                raw=read(path);cat,obj,reason=parse_raw(raw);cats.append(cat);request_times.append(raw['started_utc'])
                stored=read(path.with_name(f'parsed_seed_{seed}.json'))
                check('raw parse '+str(path.relative_to(SRC)),cat==stored['classification'])
                if obj is not None and cat!='parse_failure' and set(obj)!={'classification','evidence'}:
                    bad_schema.append(str(path.relative_to(SRC)))
                payload=raw['request'];expected_user=f"Language: {c['language']}\nFUNCTION:\n{c['code']}\nDOCUMENTATION:\n{c['document']}"
                check('raw prompt '+str(path.relative_to(SRC)),payload['messages']==[{'role':'system','content':runnerlock['prompts']['assessor_system']},{'role':'user','content':expected_user}])
                check('request settings '+str(path.relative_to(SRC)),payload['seed']==seed and payload['temperature']==.1 and payload['top_p']==.95 and payload['max_tokens']==1200 and payload['response_format']==runnerlock['prompts']['response_format'])
                r=csvlookup[(model,c['case_id'],seed)]
                check('CSV parse '+str(path.relative_to(SRC)),r['classification']==cat and int(r['reference_label'])==c['reference_label'])
                fr=raw.get('response',{}).get('choices',[{}])[0].get('finish_reason','transport_failure');finish[(model,fr)]+=1
                rawrows.append(dict(cohort=c['cohort'],case_id=c['case_id'],model=model,seed=seed,reference=c['reference_label'],classification=cat,reason=reason))
                raw_details[(model,c['case_id'],seed)]={'classification':cat,'evidence':obj.get('evidence') if obj else None}
            cnt=Counter(x for x in cats if x!='parse_failure'); winners=[k for k,v in cnt.items() if v>=2]
            consensus=winners[0] if winners else 'uncertain';votes[model,c['case_id']]=consensus
            check('consensus '+model+'/'+c['case_id'],consensus==oldvotes[model,c['case_id']]['consensus'])
    check('no extra-schema accepted responses',not bad_schema,bad_schema)
    check('lock before all assessor calls',lock['locked_utc']<min(request_times))
    csvsave('independently_parsed_assessor_runs.csv',rawrows)

    vectors={};max_pool_error=0.;chunks_checked=0
    for p in (SRC/'results/embeddings').glob('*.json'):
        e=read(p);vs=[];weights=[]
        check('lossless embedding chunks '+p.stem,''.join(c['raw_text'] for c in e['chunks'])==e['text'])
        for chunk,ref in zip(e['chunks'],e['request_paths']):
            raw=read(SRC/ref);payload=raw['request']['input'];prefix='search_query: ' if e['role']=='code' else 'search_document: '
            check('embedding payload '+ref,payload==prefix+chunk['raw_text'] and 0<chunk['content_tokens']<=384 and chunk['payload_tokens_with_specials']<=512)
            v=np.asarray(raw['response']['data'][0]['embedding'],dtype=np.float64)
            check('embedding valid '+ref,len(v)==768 and bool(np.isfinite(v).all()) and np.linalg.norm(v)>0)
            vs.append(v/np.linalg.norm(v));weights.append(chunk['content_tokens']);chunks_checked+=1
        pooled=np.average(np.asarray(vs),axis=0,weights=weights);pooled=(pooled/np.linalg.norm(pooled)).astype(np.float32)
        err=float(np.max(np.abs(pooled-np.asarray(e['vector']))));max_pool_error=max(max_pool_error,err)
        check('pooling '+p.stem,err<1e-7)
        vectors[p.stem]=np.asarray(e['vector'],dtype=np.float64)
    def score(a,b):return (1-float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))/2
    scores={r['case_id']:r for r in read(SRC/'results/scores_primary_scale.json')};score_rows=[];max_score_error=0.
    csvscores={(r['case_id'],int(r['seed'])):r for r in csv.DictReader((SRC/'results/score_case_runs.csv').open(encoding='utf-8'))}
    scale_record=read(SRC/'protocol/SCORE_SCALE_CORRECTION.json')
    summarytimes=[];summarycodeby={c['code_id']:c for c in cases}
    for cid,c in summarycodeby.items():
        for seed in protocol['summary_seeds']:
            p=SRC/'results/summaries'/cid/f'raw_seed_{seed}.json';r=read(p);summarytimes.append(r['started_utc'])
            text=r['response']['choices'][0]['message']['content'].strip();parsed=read(p.with_name(f'parsed_seed_{seed}.json'))
            check('summary parsed '+str(p.relative_to(SRC)),r['status']=='ok' and r['response']['choices'][0]['finish_reason']=='stop' and text==parsed['text'] and len(text.split())<=120)
            check('summary prompt '+str(p.relative_to(SRC)),r['request']['messages']==[{'role':'system','content':runnerlock['prompts']['summary_system']},{'role':'user','content':f"Language: {c['language']}\nFUNCTION:\n{c['code']}"}])
    check('scale correction before summaries',scale_record['recorded_utc']<min(summarytimes))
    for c in cases:
        row=scores[c['case_id']];doc=vectors[row['document_embedding']];raw=score(doc,vectors[row['code_embedding']])
        max_score_error=max(max_score_error,abs(raw-row['raw_code_score']))
        check('raw score '+c['case_id'],abs(raw-row['raw_code_score'])<1e-12)
        for seed in protocol['summary_seeds']:
            sr=row['summary_scores'][str(seed)];sc=score(doc,vectors[sr['embedding_key']]);cr=csvscores[c['case_id'],seed]
            max_score_error=max(max_score_error,abs(sc-sr['score']))
            check('summary score '+c['case_id']+'/'+str(seed),abs(sc-sr['score'])<1e-12 and abs(sc-float(cr['summary_score']))<1e-12 and float(cr['threshold'])==protocol['transfer_threshold'] and int(cr['prediction'])==int(sc>=protocol['transfer_threshold']))
            score_rows.append(dict(cohort=c['cohort'],case_id=c['case_id'],reference=c['reference_label'],seed=seed,score=sc,raw_score=raw,threshold=protocol['transfer_threshold']))
    table5=[];table6=[];diagnostics={};baselines=[]
    mapping={'N':'n','Positive':'positive','Negative':'negative','BinaryN':'substantive_predictions','TP':'TP','FP':'FP','TN':'TN','FN':'FN','AbstPositive':'abstain_positive','AbstNegative':'abstain_negative','Recall':'recall','Specificity':'specificity','Accuracy':'conservative_accuracy','BalancedAccuracy':'balanced_accuracy','Kappa':'Cohen_kappa_substantive_only','F1':'positive_F1'}
    def oldcheck(ident,m,old):
        for k,ok in mapping.items():check('old metric '+ident+'/'+k,abs(m[k]-old[ok])<1e-12 if m[k] is not None and old[ok] is not None else m[k]==old[ok])
    for cname in ['DocPrism','CASCADE']:
        subset=[c for c in cases if c['cohort']==cname];ids=[c['case_id'] for c in subset];y=[c['reference_label'] for c in subset]
        preval=sum(y)/len(y);base=metric(y,[0]*len(y));baselines.append(dict(Cohort=cname,Prevalence=preval,**base,NoSkillPopulationAP=preval,ChanceROC_AUC=.5))
        diagnostics[cname]={'prevalence':preval,'always_consistent_accuracy':base['Accuracy'],'always_consistent_balanced_accuracy':base['BalancedAccuracy'],'runs':{},'between_seed_predictions':[]}
        for model in protocol['assessor_models']:
            p=[{'consistent':0,'inconsistent':1}.get(votes[model,i],-1) for i in ids];m=metric(y,p)
            oldcheck(cname+'/'+model,m,oldmetrics['cohorts'][cname]['assessors'][model]);table5.append(dict(Cohort=cname,Model=model,**m))
        allps={}
        for name in ['Raw Nomic',*['Gemma '+str(s) for s in protocol['summary_seeds']]]:
            seed=None if name=='Raw Nomic' else int(name.split()[1])
            ss=[scores[i]['raw_code_score'] if seed is None else scores[i]['summary_scores'][str(seed)]['score'] for i in ids]
            p=[int(s>=protocol['transfer_threshold']) for s in ss];allps[name]=p
            m=metric(y,p);a,ap=auc_ap(y,ss)
            old=oldmetrics['cohorts'][cname]['score_adapter']['raw_code_Nomic'] if seed is None else next(x for x in oldmetrics['cohorts'][cname]['score_adapter']['Gemma_Nomic']['runs'] if x['seed']==seed)
            oldcheck(cname+'/'+name,m,old);check('AUC '+cname+'/'+name,abs(a-old['ROC_AUC'])<1e-12);check('AP '+cname+'/'+name,abs(ap-old['average_precision'])<1e-12)
            table6.append(dict(Cohort=cname,Representation=name,**m,ROC_AUC=a,AP=ap))
            def quant(vals):return {str(q):float(np.quantile(vals,q)) for q in [0,.25,.5,.75,.95,1]}
            diagnostics[cname]['runs'][name]={'all_score_quantiles':quant(ss),'positive_score_quantiles':quant([s for z,s in zip(y,ss) if z]),'negative_score_quantiles':quant([s for z,s in zip(y,ss) if not z]),'flagged_case_ids':[i for i,p_ in zip(ids,p) if p_],'missed_positive_ids':[i for i,p_,y_ in zip(ids,p,y) if y_ and not p_],'distance_of_nearest_score_to_threshold':min(abs(s-protocol['transfer_threshold']) for s in ss)}
        names=['Gemma '+str(s) for s in protocol['summary_seeds']]
        for ai,a in enumerate(names):
            for b in names[ai+1:]:
                changed=[i for i,pa,pb in zip(ids,allps[a],allps[b]) if pa!=pb]
                diagnostics[cname]['between_seed_predictions'].append({'a':a,'b':b,'different_predictions':len(changed),'case_ids':changed})
    csvsave('table5_independent.csv',table5);csvsave('table6_independent.csv',table6);csvsave('reference_baselines.csv',baselines);save('score_distribution_diagnostics.json',diagnostics)
    # Convenience raw diagnostic sample: every DocPrism positive and first six CASCADE false negatives.
    sample=[]
    for cname in ['DocPrism','CASCADE']:
        cs=[c for c in cases if c['cohort']==cname and c['reference_label']==1]
        if cname=='CASCADE':cs=[c for c in cs if scores[c['case_id']]['summary_scores']['1702']['score']<protocol['transfer_threshold']][:6]
        for c in cs:
            sample.append({**c,'model_consensus':{m:votes[m,c['case_id']] for m in protocol['assessor_models']},'raw_evidence':{m:raw_details[m,c['case_id'],2711] for m in protocol['assessor_models']},'summary1702':read(SRC/'results/summaries'/c['code_id']/'parsed_seed_1702.json'),'scores':scores[c['case_id']]})
    save('diagnostic_case_sample.json',sample)
    report={'scope':'Independent offline arithmetic, recorded raw responses, prompts, pooling, source-label polarity and reference fields. Does not establish human semantic correctness or rerun models.',
       'source_root':str(SRC),'script_sha256':sha(__file__),'checks':len(checks),'failed_checks':[c for c in checks if not c['passed']],
       'assessor_raw_calls':len(rawrows),'summary_raw_calls':len(summarytimes),'embedding_vectors':len(vectors),'embedding_chunks':chunks_checked,
       'maximum_pooling_absolute_error':max_pool_error,'maximum_score_absolute_error':max_score_error,'finish_counts':{m+'/'+f:n for (m,f),n in finish.items()},
       'table5':table5,'table6':table6,'baselines':baselines,'threshold':protocol['transfer_threshold'],
       'check_limits':['No human reannotation; external function labels cannot validate original PR-substitution meaning.','No new model calls or external-threshold tuning.','Raw payload provenance can be verified; historical local timestamps are not an independent third-party preregistration.','Upstream semantic labels are inherited, not corrected based on model predictions.']}
    save('EXTERNAL_INDEPENDENT_AUDIT.json',report);save('ALL_CHECKS.json',checks)
    print(json.dumps({k:report[k] for k in ['checks','failed_checks','assessor_raw_calls','summary_raw_calls','embedding_vectors','embedding_chunks','maximum_score_absolute_error','finish_counts']},indent=2))
    print(json.dumps({'table5':table5,'table6':table6,'baseline':baselines,'prediction_differences':{c:d['between_seed_predictions'] for c,d in diagnostics.items()}},indent=2))
    if report['failed_checks']:raise SystemExit(1)
if __name__=='__main__':main()
