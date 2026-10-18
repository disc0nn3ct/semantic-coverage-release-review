"""Prospectively specified paired cluster analysis; never calls a model.

Inputs are sealed development thresholds and saved locked-test predictions.
Synthetic --self-check data are software verification, not research observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiment_core import ROOT, classification, confusion, evaluate, load_json, save_json, utc

BOOTSTRAP_SEED = 20260926
BOOTSTRAP_RESAMPLES = 10_000
TEST_SEEDS = (1702, 1703, 1704)
BASELINES = ('raw_mx', 'raw_nomic', 'codebert', 'graphcodebert', 'tfidf', 'bm25')
COUNT_KEYS = ('tp', 'fp', 'tn', 'fn')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows):
    """Reject inconsistent target, score, grouping or cluster records before analysis."""
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty prediction set or duplicate case IDs')
    clusters = {}
    for r in rows:
        if r['split'] != 'test' or r['label'] not in (0, 1):
            raise ValueError('Expected binary labeled test records')
        if not r['files'] or not math.isfinite(r['score']):
            raise ValueError('Empty files or nonfinite case score')
        paths = [f['path'] for f in r['files']]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate eligible path inside case')
        if any(not math.isfinite(f['score']) for f in r['files']):
            raise ValueError('Nonfinite file score')
        if r['score'] != max(f['score'] for f in r['files']):
            raise ValueError('Case score does not equal maximum file score')
        targets = r['target_paths']
        if len(targets) != len(set(targets)) or not set(targets) <= set(paths):
            raise ValueError('Invalid or duplicate target paths')
        if bool(targets) != bool(r['label']):
            raise ValueError('Target presence disagrees with construction label')
        clusters.setdefault(r['cluster_id'], []).append(r)
    for cid, group in clusters.items():
        if len(group) != 4 or sum(r['label'] for r in group) != 2:
            raise ValueError(f'Cluster {cid} does not contain all four balanced variants')
        if len({r['project'] for r in group}) != 1:
            raise ValueError('A swap cluster crosses projects')
        bases = {}
        for r in group:
            bases.setdefault(r['base_id'], []).append(r)
        if len(bases) != 2 or any(len(v) != 2 or sum(r['label'] for r in v) != 1 for v in bases.values()):
            raise ValueError('Incomplete paired base bundle')
        for pair in bases.values():
            a, b = pair
            identity = lambda x: {(f['id'], f['path'], f['changed_lines']) for f in x['files']}
            if identity(a) != identity(b):
                raise ValueError('Sibling code inventories differ')
    return clusters


def comparable_identity(row):
    return (row['project'], row['split'], row['cluster_id'], row['base_id'], row['label'],
            tuple(sorted(row['target_paths'])),
            tuple(sorted((f['id'], f['path'], f['changed_lines']) for f in row['files'])))


def align_rows(rows, reference):
    validate_rows(rows)
    by_id = {r['id']: r for r in rows}
    if set(by_id) != {r['id'] for r in reference}:
        raise ValueError('Comparators or seeds have different case sets')
    aligned = [by_id[r['id']] for r in reference]
    if any(comparable_identity(a) != comparable_identity(b) for a, b in zip(aligned, reference)):
        raise ValueError('Case labels, target sets or eligible code differ between inputs')
    return aligned


def stratified_indices(cluster_ids, project_by_cluster, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Draw whole clusters within fixed project strata; reuse for every method/seed."""
    if resamples <= 0 or not cluster_ids:
        raise ValueError('Positive resample count and nonempty cluster IDs required')
    rng = np.random.default_rng(seed)
    blocks = []
    for project in sorted(set(project_by_cluster.values())):
        positions = np.asarray([i for i, cid in enumerate(cluster_ids) if project_by_cluster[cid] == project], dtype=np.int64)
        if not len(positions):
            raise ValueError('Empty project stratum')
        blocks.append(rng.choice(positions, size=(resamples, len(positions)), replace=True))
    return np.concatenate(blocks, axis=1)


def cluster_counts(rows, threshold, cluster_ids):
    groups = validate_rows(rows)
    if set(groups) != set(cluster_ids):
        raise ValueError('Cluster set mismatch')
    return np.asarray([[confusion(groups[cid], threshold)[k] for k in COUNT_KEYS] for cid in cluster_ids], dtype=np.int64)


def f1_from_counts(counts):
    counts = np.asarray(counts)
    tp, fp, _, fn = np.moveaxis(counts, -1, 0)
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=np.float64), where=den != 0)


def bootstrap_f1(selected_counts, comparator_counts, indices):
    """Average per-generation F1, never pool three runs into one confusion table."""
    selected_counts = np.asarray(selected_counts)
    comparator_counts = np.asarray(comparator_counts)
    if selected_counts.ndim != 3 or selected_counts.shape[-1] != 4:
        raise ValueError('Expected generation x cluster x 4 count tensor')
    if comparator_counts.shape != selected_counts.shape[1:]:
        raise ValueError('Comparator count shape differs')
    selected_per_seed = np.stack([f1_from_counts(counts[indices].sum(axis=1)) for counts in selected_counts], axis=1)
    selected_mean = selected_per_seed.mean(axis=1)
    baseline_f1 = f1_from_counts(comparator_counts[indices].sum(axis=1))
    return {'selected_per_seed_f1': selected_per_seed, 'selected_mean_f1': selected_mean,
            'comparator_f1': baseline_f1, 'difference_f1': selected_mean - baseline_f1}


def interval(values):
    lower, upper = np.quantile(np.asarray(values), [0.025, 0.975], method='linear')
    return {'lower': float(lower), 'upper': float(upper), 'level': 0.95, 'method': 'percentile; numpy linear quantile'}


def replicate_summary(evaluations):
    """Summarize actual three-run metrics without counting deterministic reuse as runs."""
    metrics = [r['metrics'] for r in evaluations]
    result = {}
    for key in sorted(set.intersection(*(set(r) for r in metrics))):
        vals = [r[key] for r in metrics]
        numeric = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        all_valid = len(numeric) == len(vals)
        result[key] = {'values': vals, 'valid_replicates': len(numeric),
                       'mean': float(np.mean(numeric)) if all_valid else None,
                       'sample_sd': float(np.std(numeric, ddof=1)) if all_valid and len(numeric) > 1 else None}
    return result


def analyze(root=ROOT, output=None):
    root = Path(root)
    destination = Path(output) if output else root / 'results/statistics'
    if (destination / 'statistical_summary.json').exists():
        raise RuntimeError('Analysis output already exists; preserve it and select a new output directory')
    lock_path = root / 'results/DEVELOPMENT_SELECTION_LOCK.json'
    lock = load_json(lock_path)
    chosen, comparator = lock['chosen_configuration'], lock['primary_comparator']
    if tuple(lock['test_seeds']) != TEST_SEEDS:
        raise ValueError('Test seeds differ from prospective design')
    if comparator != 'raw_' + chosen.split('_')[1] or comparator not in BASELINES:
        raise ValueError('Primary comparator is not the selected embedding raw-file baseline')
    input_hashes = {str(lock_path.relative_to(root)): sha256(lock_path)}
    if 'execution_seal_sha256' in lock:
        seal_path = root / 'protocol/EXECUTION_SEAL.json'
        if sha256(seal_path) != lock['execution_seal_sha256']:
            raise ValueError('Execution seal differs from the development lock')
        input_hashes[str(seal_path.relative_to(root))] = sha256(seal_path)
        seal = load_json(seal_path)
        for relative, expected in seal['files'].items():
            if sha256(root / relative) != expected:
                raise ValueError(f'Execution-sealed file changed: {relative}')
    threshold = float(lock['thresholds'][chosen]['threshold'])
    if not math.isfinite(threshold):
        raise ValueError('Invalid frozen threshold')
    sources, evaluations, reference = [], [], None
    for seed in TEST_SEEDS:
        path = root / 'results/scores' / f'{chosen}_test_{seed}.json'
        record = load_json(path)
        if (record['method'], record['split'], record['seed']) != (chosen, 'test', seed):
            raise ValueError('Prediction filename/content mismatch')
        rows = sorted(record['rows'], key=lambda r: r['id'])
        validate_rows(rows)
        if reference is None:
            reference = rows
        else:
            rows = align_rows(rows, reference)
        sources.append(rows)
        evaluations.append(evaluate(rows, threshold))
        input_hashes[str(path.relative_to(root))] = sha256(path)
    # Verify model rows cover exactly the frozen corpus test cases, not a successful subset.
    corpus_path = root / 'data/corpus.json'
    corpus = load_json(corpus_path)
    wanted = {c['id']: c for c in corpus['cases'] if c['split'] == 'test'}
    if set(wanted) != {r['id'] for r in reference}:
        raise ValueError('Predictions do not cover the entire frozen test corpus')
    for row in reference:
        c = wanted[row['id']]
        for key in ('project', 'cluster_id', 'base_id', 'label'):
            if row[key] != c[key]:
                raise ValueError(f'Corpus/prediction mismatch in {key}')
        if set(row['target_paths']) != set(c['target_paths']):
            raise ValueError('Corpus target paths changed')
    input_hashes[str(corpus_path.relative_to(root))] = sha256(corpus_path)
    clusters = validate_rows(reference)
    cluster_ids = sorted(clusters)
    project_by_cluster = {cid: rs[0]['project'] for cid, rs in clusters.items()}
    baselines, baseline_rows = {}, {}
    for method in BASELINES:
        path = root / 'results/scores' / f'{method}_test_deterministic.json'
        record = load_json(path)
        if record['method'] != method or record['split'] != 'test' or record.get('seed') is not None:
            raise ValueError('Deterministic comparator metadata mismatch')
        tau = float(lock['thresholds'][method]['threshold'])
        if not math.isfinite(tau):
            raise ValueError('Invalid comparator threshold')
        rows = align_rows(record['rows'], reference)
        baseline_rows[method] = rows
        baselines[method] = evaluate(rows, tau)
        input_hashes[str(path.relative_to(root))] = sha256(path)
    selected_counts = np.stack([cluster_counts(rows, threshold, cluster_ids) for rows in sources])
    comparator_counts = cluster_counts(baseline_rows[comparator], lock['thresholds'][comparator]['threshold'], cluster_ids)
    indices = stratified_indices(cluster_ids, project_by_cluster)
    resampled = bootstrap_f1(selected_counts, comparator_counts, indices)
    selected_point = float(np.mean([r['metrics']['f1'] for r in evaluations]))
    comparator_point = baselines[comparator]['metrics']['f1']
    per_project = {}
    for project in sorted(set(project_by_cluster.values())):
        selected = [evaluate([r for r in rows if r['project'] == project], threshold) for rows in sources]
        per_project[project] = {'case_count': sum(r['project'] == project for r in reference),
            'cluster_count': sum(v == project for v in project_by_cluster.values()),
            'selected_per_seed': dict(zip(map(str, TEST_SEEDS), selected)),
            'selected_summary': replicate_summary(selected),
            'baselines': {m: evaluate([r for r in rows if r['project'] == project], lock['thresholds'][m]['threshold'])
                          for m, rows in baseline_rows.items()}}
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'paired_cluster_bootstrap.npz', cluster_ids=np.asarray(cluster_ids),
                        sample_cluster_indices=indices, selected_cluster_confusions=selected_counts,
                        comparator_cluster_confusions=comparator_counts, **resampled)
    summary = {'created_utc': utc(), 'selected_configuration': chosen, 'primary_comparator': comparator,
        'test_seeds': list(TEST_SEEDS), 'case_count': len(reference), 'positive_case_count': sum(r['label'] for r in reference),
        'cluster_count': len(cluster_ids), 'cluster_project_map': project_by_cluster,
        'selected_per_seed': dict(zip(map(str, TEST_SEEDS), evaluations)),
        'selected_summary': replicate_summary(evaluations), 'baselines': baselines, 'per_project': per_project,
        'primary_endpoint': 'Arithmetic mean of three separately computed generation-level positive-case F1 estimates',
        'primary_contrast': {'selected_mean_f1': selected_point, 'comparator_f1': comparator_point,
                            'difference_f1': selected_point - comparator_point,
                            'selected_ci': interval(resampled['selected_mean_f1']),
                            'comparator_ci': interval(resampled['comparator_f1']),
                            'difference_ci': interval(resampled['difference_f1'])},
        'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED,
            'rng': 'numpy.random.default_rng (PCG64)', 'numpy_version': np.__version__,
            'unit': 'complete six-PR swap cluster, all four cases and all three seeds together',
            'strata': 'fixed project; original number of clusters sampled with replacement in each stratum',
            'paired': True, 'conditional_on': 'four selected projects, corpus construction and frozen artifacts',
            'p_values': 'not calculated; other comparator/project results are descriptive'},
        'limitations': ['Source-record substitution is mechanical ground truth, not independent semantic truth.',
            'Three generations provide a limited variability estimate; a seed field does not prove effective seed support.',
            'Intervals are exploratory cluster-sampling intervals conditional on these fixed projects.',
            'Statistical success does not replace independent human annotation or measured human review time.'],
        'input_sha256': input_hashes,
        'analysis_source_sha256': sha256(__file__),
        'core_source_sha256': sha256(Path(__file__).with_name('experiment_core.py')),
        'bootstrap_archive_sha256': sha256(destination / 'paired_cluster_bootstrap.npz')}
    save_json(destination / 'statistical_summary.json', summary)
    lines = ['# Locked-test statistical analysis', '',
             f"Selected configuration: `{chosen}`. Paired comparator: `{comparator}`.", '',
             f"{len(reference)} constructed cases in {len(cluster_ids)} whole swap clusters. Three fresh requested generation seeds: 1702, 1703, 1704.", '',
             '| Endpoint | Estimate | 95% project-stratified cluster percentile interval |',
             '|---|---:|---|']
    for name, point, key in [('Selected mean F1', selected_point, 'selected_mean_f1'),
                             ('Same-embedding raw-file F1', comparator_point, 'comparator_f1'),
                             ('Paired F1 difference', selected_point - comparator_point, 'difference_f1')]:
        ci = interval(resampled[key])
        lines.append(f"| {name} | {point:.6f} | [{ci['lower']:.6f}, {ci['upper']:.6f}] |")
    lines.extend(['', f"Selected F1 sample SD across the three generation estimates: {summary['selected_summary']['f1']['sample_sd']:.6f}.", '',
        'All generation-specific confusion counts, macro/micro localization, workload and file-budget results are in statistical_summary.json. Comparator reuse is deterministic, not three independent baseline runs.', '',
        'The bootstrap retains all four cluster variants and all generation outputs together. Its 10,000 resamples are not new LLM runs. Intervals are exploratory and conditional on the selected projects and construction. No p-values or unplanned superiority tests are reported. These results do not establish natural omission truth, independent human validation or reviewer-time savings.'])
    (destination / 'STATISTICAL_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def self_check():
    """Small artificial count tensors, with independently enumerable expected values."""
    ids = ['a0', 'a1', 'b0', 'b1']
    projects = {'a0': 'A', 'a1': 'A', 'b0': 'B', 'b1': 'B'}
    indices = stratified_indices(ids, projects, resamples=50, seed=BOOTSTRAP_SEED)
    assert indices.shape == (50, 4)
    assert np.all(indices[:, :2] < 2) and np.all(indices[:, 2:] >= 2)
    assert np.array_equal(indices, stratified_indices(ids, projects, 50, BOOTSTRAP_SEED))
    # Three genuinely different synthetic seed performances; mean F1 != pooled F1.
    s1 = np.tile([2, 0, 2, 0], (4, 1))
    s2 = np.tile([0, 0, 2, 2], (4, 1))
    s3 = np.tile([2, 2, 0, 0], (4, 1))
    base = np.tile([1, 0, 2, 1], (4, 1))
    result = bootstrap_f1(np.stack([s1, s2, s3]), base, indices)
    expected_mean = (1 + 0 + 2 / 3) / 3
    assert np.allclose(result['selected_mean_f1'], expected_mean)
    assert np.allclose(result['comparator_f1'], 2 / 3)
    assert np.allclose(result['difference_f1'], expected_mean - 2 / 3)
    assert not np.isclose(expected_mean, f1_from_counts(np.stack([s1, s2, s3]).sum(axis=(0, 1))))
    same = bootstrap_f1(np.stack([base] * 3), base, indices)
    assert np.all(same['difference_f1'] == 0)
    # Explicit resample, brute-force repeated cluster counts.
    draw = np.asarray([[0, 0, 2, 3]])
    variable = np.asarray([[2, 0, 2, 0], [0, 2, 0, 2], [1, 1, 1, 1], [2, 1, 1, 0]])
    out = bootstrap_f1(np.stack([variable] * 3), base, draw)
    counts = {k: int(sum(variable[i, j] for i in draw[0])) for j, k in enumerate(COUNT_KEYS)}
    assert np.isclose(out['selected_mean_f1'][0], classification(counts)['f1'])
    ci = interval(np.zeros(100))
    assert ci['lower'] == 0 and ci['upper'] == 0
    return {'status': 'passed', 'fixture': 'artificial counts only; no benchmark observations',
            'checks': ['project-stratum preservation', 'deterministic resampling', 'per-seed F1 mean instead of pooled F1',
                       'correct paired difference', 'identical-method zero difference', 'explicit duplicated-cluster arithmetic',
                       'degenerate percentile interval']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
    else:
        result = analyze(args.root, args.output)
        print(json.dumps({'status': 'complete', 'primary_contrast': result['primary_contrast']}, indent=2))
