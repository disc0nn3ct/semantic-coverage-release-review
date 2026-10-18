"""Post-result diagnostics only; no source changes, fitting, or model calls."""
from pathlib import Path
import difflib, json
from audit_external import SRC, OUT, read, save, csvsave, metric, auc_ap, sha

cases=read(SRC/'data/reference_manifest.json')['cases'];lookup={c['source_id']:c for c in cases}
scores={r['case_id']:r for r in read(SRC/'results/scores_primary_scale.json')}
lockpath=OUT/'inputs/DEVELOPMENT_SELECTION_LOCK.json'; lock=read(lockpath)
thresholds={name:lock['thresholds'][key]['threshold'] for name,key in [('selected_common','g3_nomic'),('raw_own_PR_development','raw_nomic')]}
rows=[]
for cohort in ['DocPrism','CASCADE']:
    cs=[c for c in cases if c['cohort']==cohort];y=[c['reference_label'] for c in cs];ss=[scores[c['case_id']]['raw_code_score'] for c in cs]
    for policy,tau in thresholds.items():
        m=metric(y,[int(s>=tau) for s in ss]);auc,ap=auc_ap(y,ss)
        rows.append(dict(cohort=cohort,policy=policy,threshold=tau,**m,ROC_AUC=auc,AP=ap))
csvsave('raw_threshold_sensitivity.csv',rows)
save('RAW_THRESHOLD_SENSITIVITY.json',{'analysis_status':'Post-result explanatory sensitivity; neither threshold was fitted on external reference labels. Original locally frozen common-threshold Table 6 remains unchanged.','threshold_source':'external_audit/inputs/DEVELOPMENT_SELECTION_LOCK.json','threshold_source_sha256':sha(lockpath),'rows':rows})

samples=read(OUT/'diagnostic_case_sample.json');sampleids={c['source_id'] for c in samples}
pairs=[]
for p in read(SRC/'data/reference_manifest.json')['pair_audit']:
    if p['prior_source_id'] not in sampleids:continue
    before=lookup[p['prior_source_id']];after=lookup[p['fixed_source_id']]
    pairs.append({'pair_id':p['pair_id'],'prior_case_id':before['case_id'],'fixed_case_id':after['case_id'],'source_id':before['source_id'],
      'code':before['code'],'before_document':before['document'],'after_document':after['document'],
      'diff':'\n'.join(difflib.unified_diff(before['document'].splitlines(),after['document'].splitlines(),fromfile='before',tofile='after',lineterm=''))})
save('sample_document_changes.json',pairs)
print(json.dumps({'threshold_sensitivity':rows,'sample_document_changes':pairs},indent=2))
