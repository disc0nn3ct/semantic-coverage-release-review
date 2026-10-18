"""Independent group-bootstrap replay from saved scores and group manifests."""
import json, argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--research-root',type=Path,default=Path(r'D:\study\project\research\vmfast research'),help='Extracted replication/ directory containing results/.')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent,help='Directory for new audit results; scientific inputs are never written.')
args=parser.parse_args()
R=args.research_root.resolve()
O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
lock=read(R/'results/DEVELOPMENT_SELECTION_LOCK.json')
sources=[read(R/f'results/scores/g3_nomic_test_{s}.json')['rows'] for s in (1702,1703,1704)]
sources.append(read(R/'results/scores/raw_nomic_test_deterministic.json')['rows'])
thresholds=[lock['thresholds']['g3_nomic']['threshold']]*3+[lock['thresholds']['raw_nomic']['threshold']]
full=read(R/'results/statistical_sensitivities/sensitivity_summary.json')
ex=read(R/'results/editorial_reaudit_20260930/dependency_exclusion_sensitivity.json')
report={}; failures=[]
for name,groups,truth in [('primary',full['source_groups'],full['source_group_contrast']),('exclude_shared_patch',ex['groups'],{'selected_ci':ex['selected_mean_f1_interval'],'difference_ci':ex['difference_f1_interval']})]:
    # Keep manifest order to reproduce the recorded PCG64 draw stream.
    rng=np.random.default_rng(20260926)
    choices=[]
    for project in sorted({g['project'] for g in groups}):
        slots=np.array([i for i,g in enumerate(groups) if g['project']==project])
        choices.append(rng.choice(slots,(10000,len(slots)),replace=True))
    draws=np.concatenate(choices,axis=1)
    scores=[]
    for rows,threshold in zip(sources,thresholds):
        byid={r['id']:r for r in rows}; counts=[]
        for g in groups:
            pairs=[(byid[c]['label']==1,byid[c]['score']>=threshold) for c in g['case_ids']]
            counts.append([sum(y and p for y,p in pairs),sum(not y and p for y,p in pairs),sum(y and not p for y,p in pairs)])
        summed=np.array(counts)[draws].sum(axis=1)
        tp,fp,fn=summed.T
        scores.append(2*tp/(2*tp+fp+fn))
    selected=np.mean(scores[:3],axis=0); difference=selected-scores[3]
    intervals={key:np.quantile(values,[.025,.975],method='linear').tolist() for key,values in [('selected_ci',selected),('difference_ci',difference)]}
    for key,values in intervals.items():
        if not np.allclose(values,[truth[key]['lower'],truth[key]['upper']],rtol=0,atol=1e-15):failures.append(name+' '+key)
    report[name]={'groups':len(groups),'intervals':intervals,'draws':10000}
report['failures']=failures;report['numpy_version']=np.__version__
(O/'BOOTSTRAP_RECALCULATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
if failures:raise SystemExit(1)
