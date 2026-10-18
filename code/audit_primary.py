"""Independent saved-score checks; no model calls, network or changed decisions."""
import json, math, statistics, hashlib, argparse
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--research-root',type=Path,default=Path(r'D:\study\project\research\vmfast research'),help='Extracted replication/ directory containing results/ and protocol/.')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent,help='Directory for new audit results; scientific inputs are never written.')
args=parser.parse_args()
ROOT=args.research_root.resolve()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
read=lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
lock=read(ROOT/'results/DEVELOPMENT_SELECTION_LOCK.json')
fail=[]
hashes={}
def check(ok,msg):
    if not ok: fail.append(msg)
def metrics(rows,t):
    tp=sum(r['label']==1 and r['score']>=t for r in rows)
    fp=sum(r['label']==0 and r['score']>=t for r in rows)
    tn=sum(r['label']==0 and r['score']<t for r in rows)
    fn=sum(r['label']==1 and r['score']<t for r in rows)
    hits=flags=lines=targets=0; rr=[]; cond=[]; posflags=0; falsefiles=0
    for r in rows:
        check(r['score']==max(f['score'] for f in r['files']),r['id']+' case/file score mismatch')
        found=[f for f in r['files'] if f['score']>=t]
        paths={f['path'] for f in found}; gold=set(r['target_paths'])
        h=len(paths & gold)
        hits+=h; flags+=len(paths); lines+=sum(f['changed_lines'] for f in found)
        falsefiles+=len(paths)-h
        if r['label']:
            targets+=len(gold); posflags+=len(paths); rr.append(h/len(gold))
            if r['score']>=t: cond.append(h/len(gold))
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,n=len(rows),f1=2*tp/(2*tp+fp+fn),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),specificity=tn/(tn+fp),target_hits=hits,target_denominator=targets,flagged_files=flags,positive_flagged_files=posflags,flagged_changed_lines=lines,mean_files=flags/len(rows),mean_lines=lines/len(rows),mean_nontarget_files=falsefiles/len(rows),macro_recall=statistics.mean(rr),micro_recall=hits/targets,all_case_precision=hits/flags if flags else 0,positive_precision=hits/posflags if posflags else 0,conditional_macro_recall=statistics.mean(cond) if cond else None)

methods=[('g3_nomic',1702),('g3_nomic',1703),('g3_nomic',1704)]+[(m,'deterministic') for m in ['raw_nomic','raw_mx','tfidf','bm25','codebert','graphcodebert']]
results={}; allrows={}
for m,seed in methods:
    p=ROOT/f'results/scores/{m}_test_{seed}.json'
    hashes[str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=read(p)['rows']; allrows[f'{m}_{seed}']=rows
    v=metrics(rows,lock['thresholds'][m]['threshold']); results[f'{m}_{seed}']=v
    check(v['n']==96 and v['tp']+v['fn']==48 and v['tn']+v['fp']==48 and v['target_denominator']==102,m+' denominators')
    check(abs(v['macro_recall']-v['recall']*v['conditional_macro_recall'])<1e-14,m+' macro gate')

primary=read(ROOT/'results/statistical_sensitivities/sensitivity_summary.json')
selected={k:statistics.mean(results[f'g3_nomic_{s}'][k] for s in (1702,1703,1704)) for k in ('f1','recall','specificity','macro_recall','micro_recall','all_case_precision','positive_precision','mean_files','mean_lines','mean_nontarget_files','conditional_macro_recall')}
check(abs(selected['f1']-primary['source_group_contrast']['selected_mean_f1'])<1e-15,'selected F1')
check(abs(selected['f1']-results['raw_nomic_deterministic']['f1']-primary['source_group_contrast']['difference_f1'])<1e-15,'paired difference')
selected['f1_sample_sd']=statistics.stdev(results[f'g3_nomic_{s}']['f1'] for s in (1702,1703,1704))

ex=read(ROOT/'results/editorial_reaudit_20260930/dependency_exclusion_sensitivity.json')
excluded=set(ex['excluded_test_cases']); exclusions={}
for key,rows in allrows.items():
    m,seed=key.rsplit('_',1)
    exclusions[key]=metrics([r for r in rows if r['id'] not in excluded],lock['thresholds'][m]['threshold'])
exmean=statistics.mean(exclusions[f'g3_nomic_{s}']['f1'] for s in (1702,1703,1704))
check(abs(exmean-ex['selected_mean_f1'])<1e-15,'exclusion F1')
check(abs(exmean-exclusions['raw_nomic_deterministic']['f1']-ex['difference_f1'])<1e-15,'exclusion contrast')

# Refit every development threshold and every selection fold independently.
threshold_checks={}
for method,obj in lock['thresholds'].items():
    seed=1701 if method in ('g3_mx','g3_nomic','gpt_mx','gpt_nomic') else 'deterministic'
    rows=read(ROOT/f'results/scores/{method}_dev_{seed}.json')['rows']
    candidates=sorted({r['score'] for r in rows}|{math.nextafter(max(r['score'] for r in rows),math.inf)})
    scored=[metrics(rows,t)|{'threshold':t} for t in candidates]
    best=max(scored,key=lambda x:(x['f1'],x['specificity'],-x['threshold']))
    check(best['threshold']==obj['threshold'],method+' dev threshold')
    threshold_checks[method]={'candidate_count':len(candidates),'threshold':best['threshold'],'f1':best['f1']}

oof={}
for method in lock['configuration_selection']:
    rows=read(ROOT/f'results/scores/{method}_dev_1701.json')['rows']
    counts={k:0 for k in ('tp','fp','tn','fn')}; totalflags=0
    for cluster in sorted({r['cluster_id'] for r in rows}):
        train=[r for r in rows if r['cluster_id']!=cluster]; test=[r for r in rows if r['cluster_id']==cluster]
        cand=sorted({r['score'] for r in train}|{math.nextafter(max(r['score'] for r in train),math.inf)})
        best=max((metrics(train,t)|{'threshold':t} for t in cand),key=lambda x:(x['f1'],x['specificity'],-x['threshold']))
        vv=metrics(test,best['threshold'])
        for k in counts: counts[k]+=vv[k]
        totalflags+=vv['flagged_files']
    tp,fp,tn,fn=(counts[k] for k in ('tp','fp','tn','fn'))
    oof[method]=counts|{'f1':2*tp/(2*tp+fp+fn),'specificity':tn/(tn+fp),'mean_files':totalflags/len(rows)}
    recorded=lock['configuration_selection'][method]['oof_metrics']
    check(all(oof[method][k]==recorded[k] for k in counts),method+' OOF counts')
order=lock['configuration_order']
chosen=max(oof,key=lambda m:(oof[m]['f1'],oof[m]['specificity'],-oof[m]['mean_files'],-order.index(m)))
check(chosen==lock['chosen_configuration'],'configuration winner')

report={'scope':'Independent arithmetic from saved case/file scores; no fresh inference or independent human annotation. Confidence interval numbers inspected in archived group-analysis output, not recomputed by this script.','results':results,'selected_mean':selected,'headline_contrast':primary['source_group_contrast'],'exclusion':{'cases':92,'groups':22,'selected_mean_f1':exmean,'raw_nomic_f1':exclusions['raw_nomic_deterministic']['f1'],'difference':exmean-exclusions['raw_nomic_deterministic']['f1']},'development_thresholds':threshold_checks,'oof':oof,'chosen_configuration':chosen,'failures':fail,'input_sha256':hashes}
(OUT/'PRIMARY_RECALCULATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'failures':fail,'selected':selected,'chosen':chosen,'exclusion':report['exclusion']},indent=2))
if fail: raise SystemExit(1)
