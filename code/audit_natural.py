"""Independent natural-cohort count/score/budget/bootstrap checks from saved rows."""
import json, hashlib, statistics, itertools, argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--natural-root',type=Path,default=Path(r'D:\study\project\research\vmfast research\natural_expanded_20261001'),help='Extracted natural_expansion/ directory containing FREEZE.json, data/ and results/.')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent,help='Directory for new audit results; scientific inputs are never written.')
args=parser.parse_args()
R=args.natural_root.resolve()
O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
manifest=read(R/'data/cohort_manifest.json'); frozen=read(R/'FREEZE.json'); truth=read(R/'results/summary.json')
fail=[]
def check(ok,message):
    if not ok:fail.append(message)
for p,h in frozen['evidence_files'].items():check(hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p+' frozen hash')
check(hashlib.sha256((R/'data/cohort_manifest.json').read_bytes()).hexdigest()==frozen['cohort_sha256'],'manifest hash')
rows=[read(p) for p in sorted((R/'results/rows').glob('*.json'))]
check(len(rows)==156,'row count')
check(len({(r['case_id'],r['condition'],r['seed']) for r in rows})==156,'unique row keys')
cases={c['case_id']:c for c in manifest['cases']}
check(len(cases)==26,'case count')
def budget(files,targets,k):
    hit=0.;remaining=k
    for score,part in itertools.groupby(sorted(files,key=lambda f:f['score'],reverse=True),key=lambda f:f['score']):
        tied=list(part);take=min(len(tied),remaining)
        hit+=take*sum(f['path'] in targets for f in tied)/len(tied);remaining-=take
        if remaining==0:break
    return hit/len(targets)
recomputed=[]; score_error=[]; filecomparisons=[]
for row in rows:
    target=set(cases[row['case_id']]['target_paths']); check(target==set(row['target_paths']),row['case_id']+' targets')
    doc=np.array(read(R/f'results/doc_embeddings/{row["case_id"]}_{row["condition"]}.json')['vector'],dtype=float)
    threshold=truth['threshold'];check(row['threshold']==threshold,'threshold')
    files=row['files'];flagged=[]
    for f in files:
        q=np.array(f['query_vector'],dtype=float)
        s=(1-float(np.clip(np.dot(q,doc)/(np.linalg.norm(q)*np.linalg.norm(doc)),-1,1)))/2 if f['valid_summary'] else 1.
        score_error.append(abs(s-f['score']));check((s>=threshold)==f['flagged'],'score flag')
        if s>=threshold:flagged.append(f)
    hits=sum(f['path'] in target for f in flagged);lines=sum(f['changed_lines'] for f in flagged)
    record={k:row[k] for k in ('case_id','seed','condition','repo','dependency_group','prior_outcomes_exposed')}
    record.update(detected=bool(flagged),target_hit=bool(hits),flagged_files=len(flagged),flagged_lines=lines,non_target_flagged_files=len(flagged)-hits,hit_files=hits,target_files=len(target),localization_recall=hits/len(target),localization_precision=hits/len(flagged) if flagged else 0,file_reduction=1-len(flagged)/len(files),line_reduction=1-lines/sum(f['changed_lines'] for f in files))
    for prefix,ff in [('emitted',flagged),('all_file',files)]:
        for k in [1,2,3,5,10]:record[f'{prefix}_recall_at_{k}']=budget(ff,target,k)
        curve=[budget(ff,target,k) for k in range(len(files)+1)]
        record[f'{prefix}_normalized_budget_area']=sum((a+b)/2 for a,b in zip(curve,curve[1:]))/len(files)
    for key,value in record.items():
        if key in row and isinstance(value,(float,int)):
            check(abs(float(value)-float(row[key]))<1e-12,'row '+key)
    recomputed.append(record)
check(max(score_error)<1e-12,'score errors')
def describe(rs):
    keys=truth['conditions']['before']['mean'].keys()
    return {'pairs':len({r['case_id'] for r in rs}),'seed_pair_observations':len(rs),'any_alert_count':sum(r['detected'] for r in rs),'target_hit_count':sum(r['target_hit'] for r in rs),'mean':{k:statistics.mean(float(r[k]) for r in rs) for k in keys},'micro_precision':sum(r['hit_files'] for r in rs)/sum(r['flagged_files'] for r in rs) if sum(r['flagged_files'] for r in rs) else 0.,'micro_target_recall':sum(r['hit_files'] for r in rs)/sum(r['target_files'] for r in rs)}
conditions={co:describe([r for r in recomputed if r['condition']==co]) for co in ('before','after')}
new={co:describe([r for r in recomputed if r['condition']==co and not r['prior_outcomes_exposed']]) for co in ('before','after')}
def compare(left,right,prefix):
    for k,v in left.items():
        if isinstance(v,dict):compare(v,right[k],prefix+'/'+k)
        else:check(abs(v-right[k])<1e-12,prefix+'/'+k)
compare(conditions,truth['conditions'],'conditions');compare(new,truth['new_only'],'new_only')
projects={p:{co:describe([r for r in recomputed if r['condition']==co and r['repo']==p]) for co in ('before','after')} for p in sorted({r['repo'] for r in recomputed})}
compare(projects,truth['projects'],'projects')
groups=sorted({r['dependency_group'] for r in recomputed});draws=np.random.default_rng(20261001).integers(0,len(groups),size=(10000,len(groups)))
keys=list(truth['clustered_intervals']['intervals']['before']);boot={};intervals={}
for co in ('before','after'):
    nums=[];den=[]
    for g in groups:
        ids=sorted({r['case_id'] for r in recomputed if r['dependency_group']==g})
        nums.append([sum(statistics.mean(float(r[k]) for r in recomputed if r['case_id']==cid and r['condition']==co) for cid in ids) for k in keys]);den.append(len(ids))
    boot[co]=np.array(nums)[draws].sum(axis=1)/np.array(den)[draws].sum(axis=1)[:,None]
boot['after_minus_before']=boot['after']-boot['before']
for co,bs in boot.items():
    intervals[co]={k:np.quantile(bs[:,i],[.025,.975],method='linear').tolist() for i,k in enumerate(keys)}
    for k,ci in intervals[co].items():check(np.allclose(ci,truth['clustered_intervals']['intervals'][co][k],rtol=0,atol=1e-12),co+'/'+k+' interval')
bykey={(r['case_id'],r['seed'],r['condition']):r for r in rows}
change={'target_file_seed_comparisons':0,'decreased':0,'increased':0,'unchanged':0,'downward_crossings':0,'upward_crossings':0}
for cid,c in cases.items():
    for seed in (1702,1703,1704):
        b={f['path']:f['score'] for f in bykey[(cid,seed,'before')]['files']};a={f['path']:f['score'] for f in bykey[(cid,seed,'after')]['files']}
        for target in c['target_paths']:
            change['target_file_seed_comparisons']+=1
            change['decreased']+=a[target]<b[target];change['increased']+=a[target]>b[target];change['unchanged']+=a[target]==b[target]
            change['downward_crossings']+=b[target]>=threshold and a[target]<threshold;change['upward_crossings']+=b[target]<threshold and a[target]>=threshold
compare(change,truth['target_score_changes'],'score change')
report={'scope':'Recomputed case/file metrics, every saved pooled-vector score, frozen evidence hashes, budget ties, source-group intervals and correction directions. No model calls. Raw inference/embedding envelopes are covered by prior archived replay, not rerun here.','conditions':conditions,'new_only':new,'projects':projects,'intervals':intervals,'score_changes':change,'frozen_evidence_hashes':len(frozen['evidence_files']),'rows':len(rows),'file_scores':len(score_error),'maximum_score_error':max(score_error),'groups':len(groups),'failures':fail}
(O/'NATURAL_RECALCULATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'failures':fail,'rows':len(rows),'max_error':max(score_error),'counts':{k:[v['any_alert_count'],v['target_hit_count'],v['mean']['localization_recall']] for k,v in conditions.items()},'before_recall_ci':intervals['before']['localization_recall'],'score_changes':change},indent=2))
if fail:raise SystemExit(1)
