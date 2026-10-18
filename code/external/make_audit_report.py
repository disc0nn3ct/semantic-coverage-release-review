"""Create a concise report from the independently recomputed audit files."""
from audit_external import SRC, OUT, read, save, csvsave, sha
from collections import Counter
import ast, statistics

a=read(OUT/'EXTERNAL_INDEPENDENT_AUDIT.json');d=read(OUT/'score_distribution_diagnostics.json')
cases=read(SRC/'data/reference_manifest.json')['cases']; votes={(r['model'],r['case_id']):r['consensus'] for r in read(SRC/'results/consensus_predictions.json')}
runpreds=read(SRC/'results/consensus_predictions.json');old=read(SRC/'results/EXTERNAL_RESULTS.json')
more={}
for cohort in ['DocPrism','CASCADE']:
    cs=[c for c in cases if c['cohort']==cohort];ids=[c['case_id'] for c in cs]
    av=[votes['qwen3_8_27b',i] for i in ids];bv=[votes['ministral3_14b',i] for i in ids]
    freq=Counter(av+bv);total=2*len(ids);observed=sum(x!=y for x,y in zip(av,bv))/len(ids)
    expected=sum(n*(total-n) for n in freq.values())/(total*(total-1))
    alpha=1-observed/expected
    exact=sum(x==y for x,y in zip(av,bv));runs=[r for r in a['table6'] if r['Cohort']==cohort and r['Representation'].startswith('Gemma')]
    more[cohort]={'cross_family_exact_numerator':exact,'cross_family_exact_denominator':len(ids),'nominal_alpha':alpha,
     'mean':{k:statistics.mean(r[k] for r in runs) for k in ['Recall','F1','ROC_AUC','AP']},
     'sample_SD':{k:statistics.stdev(r[k] for r in runs) for k in ['Recall','F1','ROC_AUC','AP']}}
    assert abs(alpha-old['cohorts'][cohort]['model_agreement']['nominal_alpha_all_categories'])<1e-12
    assert exact==old['cohorts'][cohort]['model_agreement']['exact_numerator']
context={'python_syntax_failures':[],'unstripped_raw_docstring_delimiters':[]}
for c in cases:
    if c['language']=='Python':
        try:ast.parse(c['code'])
        except SyntaxError as e:context['python_syntax_failures'].append({'case_id':c['case_id'],'reference':c['reference_label'],'reason':str(e)})
        if c['document'].startswith(('r"""',"r'''",'R"""',"R'''")):
            context['unstripped_raw_docstring_delimiters'].append({'case_id':c['case_id'],'source_id':c['source_id']})
more['context']=context
save('ADDITIONAL_DIAGNOSTICS.json',more)
csvsave('table5_compact.csv',[{k:r[k] for k in ['Cohort','Model','N','Positive','Negative','TP','FP','TN','FN','AbstPositive','AbstNegative','Recall','BalancedAccuracy','Kappa']} for r in a['table5']])
csvsave('table6_compact.csv',[{k:r[k] for k in ['Cohort','Representation','TP','FP','TN','FN','Recall','F1','ROC_AUC','AP']} for r in a['table6']])
report='''# Independent audit of external reference experiments

Scope: Tables 5–6 and their saved upstream labels, model requests/responses, parsing, consensus, embeddings and score calculations. The original source directories were not modified. No model calls, network requests, relabelling, threshold fitting or outcome-based sample exclusions were performed.

## Confirmed arithmetic and provenance

- All 10,588 automated checks passed. Audited every 1,164 assessor response, 369 summary response, 671 pooled embedding and 686 recorded embedding chunks. Corrected score reconstruction has maximum absolute discrepancy 0.0. Frozen source hashes, upstream code/document fields, label polarity, hidden-label prompt fields, requests, aggregation and reported values agree.
- Qwen has 575 stop responses and seven length failures; Ministral has 582 stop responses. All failures were retained. Every accepted response also satisfies the requested exact two-field JSON schema, although the historical parser itself accepts extra fields.
- The score-scale correction was recorded before summary inference; it implements `(1 - normalized cosine) / 2`. The common external threshold remains 0.1510206719265197. No count or metric in Tables 5–6 requires numerical replacement.
- Counts in Table 5 partition as TP + FN + abstention-positive = reference positives; TN + FP + abstention-negative = reference negatives. Accuracy uses all N; sensitivity and specificity retain class denominators; balanced accuracy is their mean. Kappa uses only binary predictions.
- CASCADE/Qwen has 136 binary responses and six abstentions: 25 + 43 + 2 = 70 positives; 63 + 5 + 4 = 72 negatives. Accuracy = 88/142, balanced accuracy = (25/70 + 63/72)/2, kappa is computed on 136. These are consistent definitions, but compact slash notation concealed the different denominators.

## What the distributions mean

- DocPrism has only six positives and 46 negatives. Always predicting consistent gives accuracy 46/52 = 0.8846 and balanced accuracy 0.5. Qwen's 0.8846 accuracy therefore is not evidence of strong overall detection: it finds two of six positives, while making two false alerts.
- Qwen predicts inconsistency on four of 52 DocPrism cases and 30 of 142 CASCADE cases. Ministral predicts inconsistency on 23/52 and 80/142 respectively. This observed difference in positive-call frequency explains much of the recall/false-alert trade-off; it is not a swap of TP and TN columns.
- At the common transferred threshold, Gemma flags 5/52 DocPrism functions in every run and 12/142, 12/142, 11/142 CASCADE functions. Positive recall is 1/6 for all DocPrism runs and 6/70, 6/70, 5/70 for CASCADE. The high specificity accompanies poor sensitivity.
- DocPrism has exactly the same binary predictions across all three Gemma runs. Continuous scores differ, so ranking metrics differ. CASCADE seeds 1702 and 1703 have identical aggregate confusion counts but different predictions on four cases. Seeds 1702/1704 differ on one; 1703/1704 on three. Identical confusion counts do not imply copied runs.
- Average precision must be read alongside class prevalence: 0.1154 for DocPrism and 0.4930 for CASCADE. The constant-score AP benchmark equals prevalence; this is not a confidence bound or exact finite-sample mean AP of random permutations. Mean Gemma AP is approximately 0.2961 and 0.5414 respectively.

## Material interpretive limitations found

1. The frozen assessor instruction requires an explicit behavioral contradiction and says absence alone is not a contradiction. CASCADE contains references formed by documentation clarifications. For example, C032 adds that countMatches uses non-overlapping matches, and C044 adds behavior for incomparable inputs. Responses calling the previous text consistent can reflect this narrower instruction. The experiment tests agreement between this protocol and released labels, not a perfectly aligned semantic ground truth. Do not discard these cases, reverse labels or retroactively broaden the prompt.
2. Some supplied snapshots omit the exact dependency that determines the documented behavior. C010/C012 delegate to a superclass; C042 delegates to an overload. The code/document bytes match the source artifact, but the observed input cannot establish the omitted implementation. This limits semantic interpretation of errors.
3. A diagnostic sample also shows unsupported generated details: the C020 hash64 summary describes GET_BYTES_CHARSET as the system default without its definition, and the C042 nextUniform summary states an inclusive lower bound and no exceptions without the callee. These are observed examples, not a quantified error taxonomy or proof that every low score has this cause.
4. Seven supplied Python functions do not parse standalone, all reference-negative. The existing 45-case parseability sensitivity retains all six positives and should remain clearly secondary. A single Requests/28 source string retains the raw-string delimiter r triple quotes because the cleaner handles ordinary triple quotes only. This is a literal extra token sequence, not lost documentation; the raw input should remain unchanged for this audit and the preprocessing description should be exact.
5. Published function-level reference labels cannot replace a second human assessment of the original PR-substitution cases. The results neither establish human agreement nor fully close the original annotation concern.

## Threshold fairness sensitivity

The original locally frozen external diagnostic applies the selected pipeline threshold to both representations. Raw Nomic's own previously locked PR-development threshold is 0.13821366023763004. An additional post-result sensitivity uses that existing value without fitting on the external labels:

| Cohort | Raw threshold policy | TP | FP | TN | FN | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| DocPrism | Original common threshold | 1 | 6 | 40 | 5 | 0.1667 | 0.1538 |
| DocPrism | Own PR-development threshold | 1 | 10 | 36 | 5 | 0.1667 | 0.1176 |
| CASCADE | Original common threshold | 5 | 1 | 71 | 65 | 0.0714 | 0.1316 |
| CASCADE | Own PR-development threshold | 8 | 7 | 65 | 62 | 0.1143 | 0.1882 |

AUC/AP are unchanged. This secondary check does not demonstrate successful transfer either. Keep original Table 6 unchanged, disclose that its fixed-threshold rows share a common rule, and avoid interpreting it as a comparison of separately calibrated complete pipelines.

## Concrete presentation changes

- Replace TP/FP/TN/FN slash cells with four separately labelled columns.
- Use cohort blocks whose headings state positive/negative counts. Keep abstention-positive and abstention-negative as separate columns in Table 5.
- Show recall, balanced accuracy and kappa in Table 5, with raw accuracy in the text/supplement next to its majority baseline. Explain all denominators beside the table.
- Show recall, F1, ROC AUC and AP in Table 6. Preserve per-run counts; specificity can remain in the supplement. State that seed IDs are repetitions of the same model, not different model variants.
- Call the two external cohorts reference-label diagnostics on a distinct function-level task. Explicitly state weak fixed-threshold transfer; do not attribute the entire deficit to threshold shift because AUC is also modest and task/context definitions differ.
- A suitable limitation sentence is: “The frozen assessor prompt required explicit contradiction, whereas some released references encode missing behavioral details; reference disagreement therefore also reflects construct and context mismatch.”

## Reproduction and evidence

Run `python audit_external.py <external_validation_root>`, then `python diagnose_context.py <external_validation_root>`, then `python make_audit_report.py <external_validation_root>`. The original development lock is preserved byte-for-byte under `external_audit/inputs/DEVELOPMENT_SELECTION_LOCK.json`. After extracting the external archive, use its `replication/external_gold_validation_20260930` directory as the root. Python 3.13 and NumPy are sufficient; no network or model calls are used.

- EXTERNAL_INDEPENDENT_AUDIT.json and ALL_CHECKS.json: complete recorded checks and independently recomputed table values.
- table5_independent.csv / table6_independent.csv: all conventional metrics, including precision, recall, F1, accuracy, specificity and kappa.
- table5_compact.csv / table6_compact.csv: ready-to-format main-table fields.
- score_distribution_diagnostics.json: all quantiles, flagged/missed case IDs and between-seed prediction differences.
- RAW_THRESHOLD_SENSITIVITY.json / raw_threshold_sensitivity.csv: frozen alternative-threshold sensitivity.
- diagnostic_case_sample.json / sample_document_changes.json: inspectable examples; qualitative author/agent examination, not independent human annotation.
- ADDITIONAL_DIAGNOSTICS.json: recomputed model agreement and mean/SD plus source-context counts.

Conclusion: arithmetic is reproducible within the audited scope. Presentation and the interpretation of external validation need revision. The measured low transfer performance is real under the frozen inputs and decision rule; making the manuscript scientifically stronger requires reporting that limit precisely, not making the results appear larger.
'''
(OUT/'EXTERNAL_AUDIT_REPORT.md').write_text(report,encoding='utf-8')
print('report written; additional diagnostics:',more)
