"""Supplementary source-dependence and frozen text-quality analyses.

The primary sealed statistical_analysis.py is unchanged. --record-plan reads
source manifests only. --run is for use only after all three locked runs finish
and the coordinating task authorizes test-outcome inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from experiment_core import ROOT, confusion, evaluate, load_json, save_json, utc
from statistical_analysis import (
    BASELINES, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, COUNT_KEYS, TEST_SEEDS,
    align_rows, bootstrap_f1, interval, replicate_summary, sha256,
    stratified_indices, validate_rows,
)

PLAN = 'protocol/STATISTICAL_SENSITIVITIES_PLAN_20260926.json'
GROUPS = 'data/constituent_commit_audit/source_dependency_groups.json'
QUALITY = 'protocol/QUALITY_COHORT_LOCK.json'
CORPUS = 'data/corpus.json'
LOCK = 'results/DEVELOPMENT_SELECTION_LOCK.json'


def test_group_definition(corpus, mapping):
    """Check a source-only exhaustive partition, without reading predictions."""
    cases = {c['id']: c for c in corpus['cases'] if c['split'] == 'test'}
    clusters = {}
    for case in cases.values():
        clusters.setdefault(case['cluster_id'], set()).add(case['id'])
    groups = []
    seen_cases, seen_clusters = set(), set()
    if mapping.get('cross_split_group_count') != 0:
        raise ValueError('A cross-split source dependency requires a separate validity analysis')
    for group in mapping['groups']:
        if group['splits'] != ['test']:
            if 'test' in group['splits']:
                raise ValueError('Dependency group straddles development/test')
            continue
        ids = set(group['case_ids'])
        cids = set(group['original_cluster_ids'])
        if not ids or not cids or ids & seen_cases or cids & seen_clusters:
            raise ValueError('Empty/overlapping source groups')
        if not ids <= set(cases) or not cids <= set(clusters):
            raise ValueError('Source group contains an unknown case/cluster')
        expected = set().union(*(clusters[cid] for cid in cids))
        if ids != expected or len(ids) != len(group['case_ids']):
            raise ValueError('Source group does not preserve every original cluster variant')
        projects = {cases[i]['project'] for i in ids}
        if len(projects) != 1 or projects != set(group['projects']):
            raise ValueError('Source group crosses project strata')
        if sum(cases[i]['label'] for i in ids) * 2 != len(ids):
            raise ValueError('Source group is not balanced by construction')
        groups.append({'id': group['dependency_group_id'], 'project': next(iter(projects)),
                       'case_ids': sorted(ids), 'original_cluster_ids': sorted(cids),
                       'case_count': len(ids)})
        seen_cases.update(ids)
        seen_clusters.update(cids)
    if seen_cases != set(cases) or seen_clusters != set(clusters):
        raise ValueError('Source groups omit frozen test cases/clusters')
    if len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Duplicate dependency group ID')
    return sorted(groups, key=lambda g: g['id'])


def quality_definition(corpus, lock):
    """Recompute the locked whole-cluster criterion from source body counts."""
    all_cases = {c['id']: c for c in corpus['cases']}
    units = {u['id']: u for u in corpus['units']}
    cluster_units, cluster_cases = {}, {}
    for c in all_cases.values():
        cluster_units.setdefault(c['cluster_id'], set()).update(c['unit_ids'])
        cluster_cases.setdefault(c['cluster_id'], set()).add(c['id'])
    qualifying = {cid for cid, ids in cluster_units.items()
                  if len(ids) == 6 and all(units[i]['body_words'] >= 10 for i in ids)}
    expected = set().union(*(cluster_cases[cid] for cid in qualifying)) if qualifying else set()
    if set(lock['cluster_ids']) != qualifying or set(lock['case_ids']) != expected:
        raise ValueError('Quality lock differs from the source-only >=10-body-word criterion')
    if len(lock['case_ids']) != len(set(lock['case_ids'])):
        raise ValueError('Duplicate quality case ID')
    test_ids = sorted(i for i in expected if all_cases[i]['split'] == 'test')
    test_clusters = sorted({all_cases[i]['cluster_id'] for i in test_ids})
    for cid in test_clusters:
        if not cluster_cases[cid] <= set(test_ids):
            raise ValueError('Quality subset breaks a whole original test cluster')
    projects = sorted({c['project'] for c in all_cases.values() if c['split'] == 'test'})
    return {'case_ids': test_ids, 'cluster_ids': test_clusters,
            'project_counts': {p: sum(all_cases[i]['project'] == p for i in test_ids) for p in projects},
            'rule': 'Every one of all six PRs has frozen cleaned body_words >=10',
            'refit': False, 'status': 'descriptive sensitivity; primary cohort unchanged'}


def record_plan(root=ROOT):
    root = Path(root)
    destination = root / PLAN
    if destination.exists():
        raise RuntimeError('Sensitivity plan already exists; do not silently replace its timestamp')
    corpus, mapping, quality_lock = (load_json(root / p) for p in (CORPUS, GROUPS, QUALITY))
    corpus_hash = sha256(root / CORPUS)
    if mapping['corpus_sha256'] != corpus_hash or quality_lock['corpus_sha256'] != corpus_hash:
        raise ValueError('Source maps target a different frozen corpus')
    groups = test_group_definition(corpus, mapping)
    quality = quality_definition(corpus, quality_lock)
    plan = {'recorded_utc': utc(), 'kind': 'post-seal source-only supplementary analysis specification',
        'stage': 'after inference began; no test scores loaded by this script or author while recording plan',
        'primary_analysis_unchanged': True, 'model_or_data_changes': False,
        'source_groups_generated_utc': mapping['generated_utc'],
        'group_definition': mapping['edge_rule'], 'test_group_count': len(groups),
        'group_case_sizes': [g['case_count'] for g in groups],
        'project_group_counts': {p: sum(g['project'] == p for g in groups)
                                 for p in sorted({g['project'] for g in groups})},
        'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED,
            'strata': 'project', 'unit': 'complete source-dependency group',
            'draws_per_stratum': 'original number of dependency groups in that project',
            'paired': 'all cases and three seeds retained together for every sampled group',
            'variable_case_count': True,
            'interpretation': 'Eight-case merged group remains whole; some replicates contain more/fewer than96 cases.',
            'endpoint': 'mean of three generation-level F1 values minus same-embedding deterministic raw-file F1',
            'interval': '95% percentile; NumPy linear quantile; no p-values'},
        'quality_subset': quality,
        'quality_analysis': 'All core metrics for selected three runs and six baselines; unchanged full-development thresholds; no cohort refit; no quality-subset inferential claim',
        'source_sha256': {p: sha256(root / p) for p in (CORPUS, GROUPS, QUALITY, LOCK,
                         'protocol/EXECUTION_SEAL.json', 'src/statistical_analysis.py',
                         'src/experiment_core.py', 'src/statistical_sensitivities.py')},
        'execution_gate': 'Run only after coordinator confirms all three locked generations complete and authorizes outcome inspection.'}
    save_json(destination, plan)
    return plan


def group_counts(rows, threshold, groups):
    by_id = {r['id']: r for r in rows}
    expected = {i for g in groups for i in g['case_ids']}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError('Grouped counts require exactly the full frozen test cohort')
    return np.asarray([[confusion([by_id[i] for i in g['case_ids']], threshold)[k]
                        for k in COUNT_KEYS] for g in groups], dtype=np.int64)


def exact_subset(rows, wanted_ids):
    by_id = {r['id']: r for r in rows}
    wanted = set(wanted_ids)
    if len(by_id) != len(rows) or len(wanted_ids) != len(wanted) or not wanted <= set(by_id):
        raise ValueError('Invalid/missing/duplicate quality-case membership')
    chosen = [by_id[i] for i in sorted(wanted)]
    if {r['id'] for r in chosen} != wanted or len(chosen) != len(wanted):
        raise AssertionError('Subset selection changed membership')
    return chosen


def analyze(root=ROOT, output=None):
    root = Path(root)
    destination = Path(output) if output else root / 'results/statistical_sensitivities'
    if destination.exists():
        raise RuntimeError('Output directory already exists; preserve it and use a new destination')
    plan = load_json(root / PLAN)
    for relative, expected in plan['source_sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Sensitivity-plan source changed: {relative}')
    seal = load_json(root / 'protocol/EXECUTION_SEAL.json')
    for relative, expected in seal['files'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Primary sealed file changed: {relative}')
    lock = load_json(root / LOCK)
    chosen, comparator = lock['chosen_configuration'], lock['primary_comparator']
    if tuple(lock['test_seeds']) != TEST_SEEDS or comparator != 'raw_' + chosen.split('_')[1]:
        raise ValueError('Selected seeds/comparator differ from frozen decision')
    # Existence/metadata check precedes reading performance. Missing runs are never omitted.
    score_paths = [f'results/scores/{chosen}_test_{seed}.json' for seed in TEST_SEEDS]
    score_paths += [f'results/scores/{m}_test_deterministic.json' for m in BASELINES]
    generation_paths = [f'results/summaries/{chosen.split("_")[0]}_test_{seed}.json' for seed in TEST_SEEDS]
    missing = [p for p in score_paths + generation_paths if not (root / p).is_file()]
    if missing:
        raise RuntimeError('All three runs and comparators must complete first: ' + ', '.join(missing))
    for seed, relative in zip(TEST_SEEDS, generation_paths):
        record = load_json(root / relative)
        if record['seed'] != seed or record['split'] != 'test' or not record.get('finished_utc'):
            raise ValueError('Generation completion metadata missing or inconsistent')
    corpus = load_json(root / CORPUS)
    mapping = load_json(root / GROUPS)
    groups = test_group_definition(corpus, mapping)
    quality = quality_definition(corpus, load_json(root / QUALITY))
    if quality != plan['quality_subset']:
        raise ValueError('Frozen quality subset changed')
    sources, evals, reference = [], [], None
    tau = float(lock['thresholds'][chosen]['threshold'])
    for seed in TEST_SEEDS:
        record = load_json(root / f'results/scores/{chosen}_test_{seed}.json')
        if (record['method'], record['split'], record['seed']) != (chosen, 'test', seed):
            raise ValueError('Selected prediction metadata mismatch')
        rows = sorted(record['rows'], key=lambda r: r['id'])
        validate_rows(rows)
        if reference is None:
            reference = rows
        else:
            rows = align_rows(rows, reference)
        sources.append(rows)
        evals.append(evaluate(rows, tau))
    expected = {c['id']: c for c in corpus['cases'] if c['split'] == 'test'}
    if set(expected) != {r['id'] for r in reference}:
        raise ValueError('Selected predictions omit frozen cases')
    for row in reference:
        c = expected[row['id']]
        if any(c[k] != row[k] for k in ('project', 'cluster_id', 'base_id', 'label')):
            raise ValueError('Source case identity mismatch')
        if set(c['target_paths']) != set(row['target_paths']):
            raise ValueError('Source target set mismatch')
    baseline_rows, baseline_evals = {}, {}
    for method in BASELINES:
        record = load_json(root / f'results/scores/{method}_test_deterministic.json')
        if record['method'] != method or record['split'] != 'test' or record.get('seed') is not None:
            raise ValueError('Comparator prediction metadata mismatch')
        baseline_rows[method] = align_rows(record['rows'], reference)
        baseline_evals[method] = evaluate(baseline_rows[method], lock['thresholds'][method]['threshold'])
    selected_counts = np.stack([group_counts(rows, tau, groups) for rows in sources])
    comparator_counts = group_counts(baseline_rows[comparator], lock['thresholds'][comparator]['threshold'], groups)
    group_ids = [g['id'] for g in groups]
    project_by_group = {g['id']: g['project'] for g in groups}
    sampled = stratified_indices(group_ids, project_by_group, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    boot = bootstrap_f1(selected_counts, comparator_counts, sampled)
    group_case_sizes = np.asarray([g['case_count'] for g in groups], dtype=np.int64)
    sample_case_counts = group_case_sizes[sampled].sum(axis=1)
    # Count tensor conservation and paired denominators are independent of performance.
    assert np.array_equal(selected_counts[0].sum(axis=1), group_case_sizes)
    assert np.all(selected_counts.sum(axis=2) == group_case_sizes[None, :])
    assert np.array_equal(comparator_counts.sum(axis=1), group_case_sizes)
    assert np.all(sample_case_counts % 2 == 0)
    point_selected = float(np.mean([e['metrics']['f1'] for e in evals]))
    point_baseline = baseline_evals[comparator]['metrics']['f1']
    quality_selected = [evaluate(exact_subset(rows, quality['case_ids']), tau) for rows in sources]
    quality_baselines = {m: evaluate(exact_subset(rows, quality['case_ids']), lock['thresholds'][m]['threshold'])
                         for m, rows in baseline_rows.items()}
    quality_projects = {}
    for project, count in quality['project_counts'].items():
        if count == 0:
            quality_projects[project] = {'case_count': 0, 'metrics': None,
                'reason': 'No whole test cluster satisfies the locked body-length criterion'}
            continue
        ids = [i for i in quality['case_ids'] if expected[i]['project'] == project]
        se = [evaluate(exact_subset(rows, ids), tau) for rows in sources]
        quality_projects[project] = {'case_count': len(ids),
            'selected_per_seed': dict(zip(map(str, TEST_SEEDS), se)),
            'selected_summary': replicate_summary(se),
            'baselines': {m: evaluate(exact_subset(rows, ids), lock['thresholds'][m]['threshold'])
                          for m, rows in baseline_rows.items()}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'source_group_bootstrap.npz',
        group_ids=np.asarray(group_ids), group_case_sizes=group_case_sizes,
        sampled_group_indices=sampled, sample_case_counts=sample_case_counts,
        selected_group_confusions=selected_counts, comparator_group_confusions=comparator_counts, **boot)
    input_paths = set(score_paths + generation_paths + list(plan['source_sha256']) + [PLAN])
    source_report = 'data/constituent_commit_audit/SOURCE_AUDIT_REPORT.md'
    if (root / source_report).exists():
        input_paths.add(source_report)
    snapshot = {}
    for relative in sorted(input_paths):
        src, dst = root / relative, destination / 'inputs' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        h = sha256(src)
        if sha256(dst) != h:
            raise AssertionError('Lossless snapshot verification failed')
        snapshot[relative] = {'sha256': h, 'bytes': src.stat().st_size,
                              'snapshot_path': str(dst.relative_to(destination)).replace('\\', '/')}
    summary = {'created_utc': utc(), 'analysis': 'supplementary; original primary24-cluster outputs retained',
        'plan_recorded_utc': plan['recorded_utc'], 'selected_configuration': chosen,
        'primary_comparator': comparator, 'test_seeds': list(TEST_SEEDS),
        'full_case_count': len(reference), 'original_test_cluster_count': len({r['cluster_id'] for r in reference}),
        'source_group_count': len(groups), 'source_groups': groups,
        'source_group_contrast': {'selected_mean_f1': point_selected, 'comparator_f1': point_baseline,
            'difference_f1': point_selected - point_baseline,
            'selected_ci': interval(boot['selected_mean_f1']),
            'comparator_ci': interval(boot['comparator_f1']), 'difference_ci': interval(boot['difference_f1'])},
        'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED,
            'generator': 'numpy.random.default_rng (PCG64)', 'numpy_version': np.__version__,
            'unit': 'source-linked original clusters, kept as whole groups',
            'sampling': 'with replacement, original dependency-group count within each project',
            'variable_case_count': {'minimum_observed': int(sample_case_counts.min()),
                'maximum_observed': int(sample_case_counts.max()), 'mean_observed': float(sample_case_counts.mean()),
                'frequency': {str(int(v)): int((sample_case_counts == v).sum()) for v in np.unique(sample_case_counts)}},
            'interpretation': 'Eight-case linked group remains intact; case denominators may vary by resample. Same group draw for methods and all three generations.',
            'interval_status': 'exploratory source-dependence sensitivity; no p-values'},
        'full_selected_per_seed': dict(zip(map(str, TEST_SEEDS), evals)),
        'full_selected_summary': replicate_summary(evals),
        'quality': {'definition': quality, 'selected_per_seed': dict(zip(map(str, TEST_SEEDS), quality_selected)),
            'selected_summary': replicate_summary(quality_selected), 'baselines': quality_baselines,
            'per_project': quality_projects, 'thresholds_unchanged': {m: lock['thresholds'][m]['threshold']
                for m in [chosen] + list(BASELINES)},
            'limitations': 'Only Airflow/Home Assistant qualify in this corpus; composition changes with body length. Descriptive only, no refit or independent semantic-validation claim.'},
        'inputs': snapshot, 'bootstrap_archive_sha256': sha256(destination / 'source_group_bootstrap.npz'),
        'limitations': ['Source audit occurred after inference began and before its agent read test metrics.',
            '501 emitted commit identities are distinct;16 head gaps and untraversed Git history limit completeness.',
            'Signed-line similarity catches a confirmed backport but not all semantic clones.',
            'Source grouping changes uncertainty units only, not corpus membership, labels, thresholds or predictions.']}
    save_json(destination / 'sensitivity_summary.json', summary)
    ci = summary['source_group_contrast']['difference_ci']
    report = [
        '# Supplementary source-dependence and text-quality analyses', '',
        f"Original primary cohort remains {len(reference)} cases / {summary['original_test_cluster_count']} construction clusters.",
        f"Source sensitivity resamples {len(groups)} dependency groups; all cases remain in point estimates.", '',
        f"Selected mean F1: {point_selected:.6f}; comparator F1: {point_baseline:.6f}; paired difference: {point_selected-point_baseline:.6f}.",
        f"Source-group percentile95% interval for difference: [{ci['lower']:.6f}, {ci['upper']:.6f}].", '',
        'The linked Keycloak group contains eight cases. Each project retains its original number of source groups per bootstrap draw; the number of cases therefore varies. All conditions and all three generations remain together, and every comparator uses the same draw.',
        f"Observed bootstrap case-count range: {sample_case_counts.min()}–{sample_case_counts.max()}; no p-values are calculated.", '',
        f"Locked quality subset: {len(quality['case_ids'])} cases / {len(quality['cluster_ids'])} original clusters.",
        'All metrics use unchanged full-development thresholds. Absent projects are reported as absent, not zero performance. The quality subset changes project composition and supports only descriptive comparison.', '',
        'The original sealed24-cluster analysis is preserved. All input files used here are copied byte-for-byte under inputs/ and hashed; raw sampled group indices and count tensors are in source_group_bootstrap.npz. This is an explicitly supplementary post-seal source audit, not a replacement primary test.']
    (destination / 'SENSITIVITY_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return summary


def self_check():
    # Synthetic balanced construction, with2 original clusters merged in one project.
    groups = [{'id': 'A-merged', 'project': 'A', 'case_ids': [f'a{i}' for i in range(8)], 'case_count': 8},
              {'id': 'A-single', 'project': 'A', 'case_ids': [f'b{i}' for i in range(4)], 'case_count': 4},
              {'id': 'B-single', 'project': 'B', 'case_ids': [f'c{i}' for i in range(4)], 'case_count': 4}]
    rows = []
    for g in groups:
        for j, cid in enumerate(g['case_ids']):
            rows.append({'id': cid, 'label': j % 2, 'score': .8 if j % 2 else .2})
    counts = group_counts(rows, .5, groups)
    assert np.array_equal(counts, [[4, 0, 4, 0], [2, 0, 2, 0], [2, 0, 2, 0]])
    indices = stratified_indices([g['id'] for g in groups], {g['id']: g['project'] for g in groups}, 1000, BOOTSTRAP_SEED)
    sizes = np.asarray([8, 4, 4])
    case_counts = sizes[indices].sum(axis=1)
    assert set(case_counts) == {12, 16, 20}
    assert np.all(indices[:, :2] < 2) and np.all(indices[:, 2] == 2)
    result = bootstrap_f1(np.stack([counts] * 3), counts, indices)
    assert np.all(result['difference_f1'] == 0)
    assert np.all(result['selected_mean_f1'] == 1)
    selected = exact_subset(rows, ['a0', 'a1', 'c0', 'c1'])
    assert {r['id'] for r in selected} == {'a0', 'a1', 'c0', 'c1'}
    assert not {'a2', 'b0'} & {r['id'] for r in selected}
    try:
        exact_subset(rows, ['missing'])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing subset member not rejected')
    try:
        group_counts(rows[:-1], .5, groups)
    except ValueError:
        pass
    else:
        raise AssertionError('Dropped source-group member not rejected')
    return {'status': 'passed', 'fixture': 'artificial only; no test outcome files read',
            'checks': ['merged8-case group retained', 'project-stratified whole-group draws',
                'variable case denominators12/16/20', 'identical-method zero paired difference',
                'exact quality subset excludes all nonmembers', 'missing subset member rejected',
                'incomplete source-group coverage rejected']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true')
    mode.add_argument('--record-plan', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--confirm-three-runs-complete', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
    elif args.record_plan:
        p = record_plan(args.root)
        print(json.dumps({'recorded_utc': p['recorded_utc'], 'test_groups': p['test_group_count'],
                          'quality_cases': len(p['quality_subset']['case_ids']),
                          'test_metrics_read': False}, indent=2))
    else:
        if not args.confirm_three_runs_complete:
            parser.error('--run requires --confirm-three-runs-complete after coordinator authorization')
        result = analyze(args.root, args.output)
        print(json.dumps({'status': 'complete', 'source_groups': result['source_group_count'],
                          'quality_cases': len(result['quality']['definition']['case_ids']),
                          'source_group_contrast': result['source_group_contrast']}, indent=2))
