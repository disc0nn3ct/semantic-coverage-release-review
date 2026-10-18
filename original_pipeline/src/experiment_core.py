"""Fixed file-level representations and evaluation utilities for the new study."""
from __future__ import annotations
import collections, hashlib, json, math, pathlib, re
import numpy as np
from local_inference import ROOT, digest, request, summarize, utc

def load_json(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))
def save_json(path,x):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def byte_chunks(text,limit=4000):
    """Lossless contiguous Unicode chunks with <=4000 UTF-8 bytes; favor line boundaries."""
    result=[]; current=''
    for line in text.splitlines(keepends=True):
        if len((current+line).encode())<=limit:
            current+=line;continue
        if current:result.append(current);current=''
        while len(line.encode())>limit:
            end=limit
            while len(line[:end].encode())>limit:end-=1
            result.append(line[:end]);line=line[end:]
        current=line
    if current:result.append(current)
    assert ''.join(result)==text
    return result

def items(corpus,split):
    accepted={'dev','development'} if split=='dev' else {split}
    cases=[c for c in corpus['cases'] if c['split'] in accepted]
    if not cases:raise ValueError(f'No cases for split {split}')
    ids={u for c in cases for u in c['unit_ids']+c['document_unit_ids']}
    docs={};files={}; mapping={}
    for unit in corpus['units']:
        if unit['id'] not in ids:continue
        docs[unit['id']]=unit['documentation']
        mapping[unit['id']]=[]
        for f in unit['files']:
            fid=unit['id']+'::'+f['path']
            files[fid]={'unit_id':unit['id'],**f,'text':'file: '+f['path']+'\n'+f['patch']}
            mapping[unit['id']].append(fid)
    return cases,docs,files,mapping

def generate(corpus,split,model_key,model_identifier,seed):
    cases,docs,files,mapping=items(corpus,split)
    outputs={}; start=utc()
    for i,(fid,f) in enumerate(sorted(files.items())):
        chunks=byte_chunks(f['text'])
        generated=[];valid=True; metadata=[]
        for j,chunk in enumerate(chunks):
            name=digest({'file_id':fid,'chunk':j})
            path=ROOT/'results'/'generations'/model_key/str(seed)/(name+'.json')
            result=summarize(model_identifier,chunk,seed,path,max_tokens=2048 if model_key=='gpt' else 768)
            valid=valid and result['valid'];generated.append(result['text']);metadata.append(result)
        outputs[fid]={'text':'\n'.join(generated),'valid':valid,'chunks':metadata,
                      'input_sha256':hashlib.sha256(f['text'].encode()).hexdigest()}
        if i%10==0 or i+1==len(files):print(f'{model_key} seed={seed} {split} {i+1}/{len(files)} files',flush=True)
    result={'model':model_key,'identifier':model_identifier,'seed':seed,'split':split,
            'started_utc':start,'finished_utc':utc(),'files':outputs}
    save_json(ROOT/'results'/'summaries'/f'{model_key}_{split}_{seed}.json',result)
    return result

def tokens(text):
    text=re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',text)
    return re.findall(r'[a-z][a-z0-9]{1,}',text.lower().replace('_',' '))

class Lexical:
    def __init__(self,dev_docs,kind):
        self.kind=kind;counts=[collections.Counter(tokens(t)) for t in dev_docs]
        self.n=len(counts);df=collections.Counter(k for c in counts for k in c)
        self.idf={k:(math.log((self.n+1)/(v+1))+1 if kind=='tfidf' else
                     math.log(1+(self.n-v+.5)/(v+.5))) for k,v in df.items()}
        self.avglen=sum(sum(c.values()) for c in counts)/max(1,self.n)
    def vector(self,text):
        c=collections.Counter(tokens(text));v={k:(1+math.log(n))*self.idf[k] for k,n in c.items() if k in self.idf}
        norm=math.sqrt(sum(x*x for x in v.values()))
        return {k:x/norm for k,x in v.items()} if norm else {}
    def coverage(self,code,doc):
        if self.kind=='tfidf':
            a,b=self.vector(code),self.vector(doc)
            return min(1.,max(0.,sum(x*b.get(k,0) for k,x in a.items())))
        # The code's distinct terms are a query; the PR description is a document.
        c=collections.Counter(tokens(doc));length=sum(c.values());score=0.
        for k in set(tokens(code)):
            tf=c.get(k,0)
            if tf and k in self.idf:
                score+=self.idf[k]*tf*2.2/(tf+1.2*(.25+.75*length/max(1,self.avglen)))
        return score/(1+score)

def score_cases(cases,files,mapping,similarity,failed=None):
    out=[]; failed=failed or set()
    for case in cases:
        scores=[]
        for uid in case['unit_ids']:
            for fid in mapping[uid]:
                s=1. if fid in failed else float(1-max(similarity(fid,did) for did in case['document_unit_ids']))
                scores.append({'id':fid,'path':files[fid]['path'],'score':s,
                               'changed_lines':files[fid]['changed_lines'],'failed_summary':fid in failed})
        scores.sort(key=lambda x:(-x['score'],x['path']))
        assert len({x['path'] for x in scores})==len(scores)
        out.append({k:case[k] for k in ('id','project','split','cluster_id','base_id','label','target_paths')}|
                   {'score':max(x['score'] for x in scores),'files':scores})
    return out

def confusion(rows,tau):
    tp=fp=tn=fn=0
    for r in rows:
        p=r['score']>=tau
        tp+=bool(p and r['label']);fp+=bool(p and not r['label'])
        tn+=bool(not p and not r['label']);fn+=bool(not p and r['label'])
    return dict(tp=tp,fp=fp,tn=tn,fn=fn)

def classification(counts):
    tp,fp,tn,fn=(counts[k] for k in ('tp','fp','tn','fn'))
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    specificity=tn/(tn+fp) if tn+fp else 0.
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return counts|dict(precision=precision,recall=recall,specificity=specificity,
         f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,accuracy=(tp+tn)/(tp+fp+tn+fn),
         balanced_accuracy=(recall+specificity)/2,mcc=(tp*tn-fp*fn)/den if den else 0.)

def calibrate(rows):
    # Every distinct >= partition is enumerated exactly, including flag none/all.
    values=sorted(set(float(r['score']) for r in rows))
    candidates=values+[math.nextafter(values[-1],math.inf)]
    sweep=[]
    for tau in candidates:
        m=classification(confusion(rows,tau));sweep.append({'threshold':tau,**m})
    best=max(sweep,key=lambda r:(r['f1'],r['specificity'],-r['threshold']))
    return best,sweep

def evaluate(rows,tau):
    m=classification(confusion(rows,tau)); positives=[r for r in rows if r['label']]
    records=[]
    for r in rows:
        target=set(r['target_paths']); flagged=[f for f in r['files'] if f['score']>=tau]
        hit=sum(f['path'] in target for f in flagged)
        record={'id':r['id'],'positive':bool(r['label']),'detected':r['score']>=tau,
           'flagged_files':len(flagged),'flagged_changed_lines':sum(f['changed_lines'] for f in flagged),
           'false_alert_files':sum(f['path'] not in target for f in flagged),'hit_files':hit,'target_files':len(target),
           'loc_precision':hit/len(flagged) if flagged else 0.,'loc_recall':hit/len(target) if target else None}
        for budget in (1,2,3,5,10):
            for prefix,ranked in (('ranked',r['files']),('flagged',flagged)):
                hits,anyhit=tied_budget(ranked,target,budget)
                record[f'{prefix}_target_recall_at_{budget}']=hits/len(target) if target else None
                record[f'{prefix}_hit_at_{budget}']=anyhit if target else None
                low,high=tied_budget_bounds(ranked,target,budget)
                record[f'{prefix}_target_recall_at_{budget}_min']=low/len(target) if target else None
                record[f'{prefix}_target_recall_at_{budget}_max']=high/len(target) if target else None
        records.append(record)
    pos=[r for r in records if r['positive']];det=[r for r in pos if r['detected']]
    avg=lambda key,rr:sum(r[key] for r in rr)/len(rr) if rr else None
    rankkeys=[f'{prefix}_{metric}_at_{b}' for prefix in ('ranked','flagged')
              for metric in ('target_recall','hit') for b in (1,2,3,5,10)]
    rankkeys += [f'{prefix}_target_recall_at_{b}_{bound}' for prefix in ('ranked','flagged')
                 for b in (1,2,3,5,10) for bound in ('min','max')]
    for key in ['loc_precision','loc_recall']+rankkeys:
        m[key]=avg(key,pos)
    m['conditional_loc_precision']=avg('loc_precision',det);m['conditional_loc_recall']=avg('loc_recall',det)
    m['micro_loc_precision']=sum(r['hit_files'] for r in pos)/sum(r['flagged_files'] for r in pos) if sum(r['flagged_files'] for r in pos) else 0.
    m['micro_loc_recall']=sum(r['hit_files'] for r in pos)/sum(r['target_files'] for r in pos)
    m['all_case_micro_loc_precision']=sum(r['hit_files'] for r in records)/sum(r['flagged_files'] for r in records) if sum(r['flagged_files'] for r in records) else 0.
    m['conditional_micro_loc_precision']=sum(r['hit_files'] for r in det)/sum(r['flagged_files'] for r in det) if sum(r['flagged_files'] for r in det) else None
    m['conditional_micro_loc_recall']=sum(r['hit_files'] for r in det)/sum(r['target_files'] for r in det) if det else None
    for key in ('hit_files','target_files','flagged_files','false_alert_files','flagged_changed_lines'):
        m[key+'_total']=sum(r[key] for r in records)
    for key in ('flagged_files','flagged_changed_lines','false_alert_files'):
        m[key+'_mean']=avg(key,records);m[key+'_positive_mean']=avg(key,pos)
        m[key+'_control_mean']=avg(key,[r for r in records if not r['positive']])
    m['summary_failed_files']=len({f['id'] for r in rows for f in r['files'] if f['failed_summary']})
    m['unique_files']=len({f['id'] for r in rows for f in r['files']})
    return {'threshold':tau,'metrics':m,'case_metrics':records}

def tied_budget(ranked,target,budget):
    """Expected target count and probability of any hit under uniform exact-score ties."""
    hits=0.;nohit=1.;remaining=budget;i=0
    while i<len(ranked) and remaining>0:
        j=i+1
        while j<len(ranked) and ranked[j]['score']==ranked[i]['score']:j+=1
        block=ranked[i:j];n=len(block);t=sum(f['path'] in target for f in block);take=min(remaining,n)
        hits+=take*t/n
        nohit*=math.comb(n-t,take)/math.comb(n,take) if n-t>=take else 0.
        remaining-=take;i=j
    return hits,1-nohit

def tied_budget_bounds(ranked,target,budget):
    low=high=0;remaining=budget;i=0
    while i<len(ranked) and remaining>0:
        j=i+1
        while j<len(ranked) and ranked[j]['score']==ranked[i]['score']:j+=1
        n=j-i;t=sum(f['path'] in target for f in ranked[i:j]);take=min(remaining,n)
        low+=max(0,take-(n-t));high+=min(t,take);remaining-=take;i=j
    return low,high
