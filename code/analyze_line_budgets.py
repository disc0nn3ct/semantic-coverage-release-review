"""Offline, post-result changed-line-cap diagnostic; only Python standard library.

Inputs are the saved original controlled-test file scores and development lock.
The adjacent protocol was fixed before this diagnostic was first calculated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
from statistics import mean, stdev


HERE = Path(__file__).resolve().parent
METHODS = ('g3_nomic', 'raw_nomic', 'raw_mx', 'codebert', 'graphcodebert', 'tfidf', 'bm25')
SEEDS = (1702, 1703, 1704)
BUDGETS = (0, 10, 25, 50, 100, 200)
POLICIES = ('emitted_prefix', 'all_file_prefix')
PROTOCOL_SHA256 = 'fcbcf2fbf1554585ea22d68b42fb6d1dad34e23d0659738a45cc0f066938234f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def ordered_files(files, threshold, policy):
    if policy not in POLICIES:
        raise ValueError(policy)
    included = files if policy == 'all_file_prefix' else [f for f in files if f['score'] >= threshold]
    return sorted(included, key=lambda f: (-f['score'], f['path'], f['id']))


def inspect_prefix(files, budget):
    if budget < 0:
        raise ValueError('Negative budget')
    used = 0
    result = []
    for f in files:
        cost = f['changed_lines']
        if not isinstance(cost, int) or cost <= 0:
            raise ValueError('Whole-file changed-line costs must be positive integers')
        if used + cost > budget:
            break
        used += cost
        result.append(f)
    return result, used


def independently_verify_prefix(files, budget, selected, used):
    """Independent cumulative-cost formulation, without calling the loop policy."""
    cumulative = list(accumulate(f['changed_lines'] for f in files))
    length = sum(total <= budget for total in cumulative)
    assert [f['id'] for f in selected] == [f['id'] for f in files[:length]]
    assert used == (cumulative[length - 1] if length else 0)
    assert used <= budget


def self_check():
    # Synthetic fixtures verify the policy, not any research observation.
    f = lambda path, cost, score: dict(id=path, path=path, changed_lines=cost, score=score)
    files = [f('expensive', 11, .9), f('target', 1, .8)]
    assert inspect_prefix(files, 10) == ([], 0)  # Cannot skip the first file.
    assert [x['id'] for x in inspect_prefix(files, 11)[0]] == ['expensive']
    assert len(inspect_prefix(files, 12)[0]) == 2
    assert inspect_prefix(files, 0) == ([], 0)
    tied = [f('z', 1, .5), f('a', 3, .5), f('b', 1, .499999999999)]
    assert [x['id'] for x in ordered_files(tied, .5, 'emitted_prefix')] == ['a', 'z']
    assert inspect_prefix(ordered_files(tied, .5, 'emitted_prefix'), 2) == ([], 0)
    assert len(ordered_files(tied, .5, 'all_file_prefix')) == 3
    no_alert = ordered_files([f('target', 1, .4)], .5, 'emitted_prefix')
    targets = {'target'}
    assert len({x['path'] for x in inspect_prefix(no_alert, 200)[0]} & targets) / len(targets) == 0
    for fs in (files, sorted(tied, key=lambda x: (-x['score'], x['path']))):
        for budget in range(16):
            selected, used = inspect_prefix(fs, budget)
            independently_verify_prefix(fs, budget, selected, used)
    return ['zero budget', 'exact-cost boundary', 'no skipping expensive files',
            'inclusive threshold', 'lexicographic exact-score ties',
            'all-positive denominator despite no alert', 'distinct all-file policy',
            'independent cumulative-cost cross-check']


def case_identity(row):
    return (row['project'], row['cluster_id'], row['base_id'], row['split'], row['label'],
            tuple(sorted(row['target_paths'])),
            tuple(sorted((f['id'], f['path'], f['changed_lines']) for f in row['files'])))


def dump_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=Path, help='Extracted replication root containing data and results')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = HERE / 'LINE_BUDGET_PROTOCOL.json'
    protocol = load(protocol_path)
    assert sha(protocol_path) == PROTOCOL_SHA256, 'Frozen diagnostic protocol was modified'
    assert protocol['budgets_changed_lines'] == list(BUDGETS)
    assert protocol['methods'] == list(METHODS)
    assert protocol['generation_seeds'] == list(SEEDS)
    input_paths = [Path('data/corpus.json'), Path('results/DEVELOPMENT_SELECTION_LOCK.json')]
    run_specs = [(method, seed, Path('results/scores') / f'{method}_test_{seed}.json')
                 for method in METHODS for seed in (SEEDS if method == 'g3_nomic' else ('deterministic',))]
    input_paths.extend(p for _, _, p in run_specs)
    input_hashes = {p.as_posix(): sha(root / p) for p in input_paths}
    # This timestamp/hash receipt is written before reading scores or computing new metrics.
    receipt = {'run_started_utc': datetime.now(timezone.utc).isoformat(),
               'protocol_sha256': sha(protocol_path), 'script_sha256': sha(Path(__file__)),
               'source_sha256': input_hashes,
               'status': 'Replay of a post-result descriptive diagnostic; no inference or network.'}
    save(out / 'RUN_INPUT_RECEIPT.json', receipt)
    tests = self_check()
    corpus = load(root / 'data/corpus.json')
    units = {u['id']: u for u in corpus['units']}
    source_files = {f"{u['id']}::{f['path']}": f for u in corpus['units'] for f in u['files']}
    test_cases = {c['id']: c for c in corpus['cases'] if c['split'] == 'test'}
    assert len(test_cases) == 96
    lock = load(root / 'results/DEVELOPMENT_SELECTION_LOCK.json')
    assert lock['chosen_configuration'] == 'g3_nomic'
    reference = None
    all_case_rows = []
    run_metadata = []
    unique_test_files = set()
    for method, seed, relative in run_specs:
        payload = load(root / relative)
        rows = payload['rows']
        assert len(rows) == 96 and len({r['id'] for r in rows}) == 96
        assert {r['id'] for r in rows} == set(test_cases)
        assert sum(r['label'] for r in rows) == 48
        assert len({r['cluster_id'] for r in rows}) == 24
        assert Counter(r['project'] for r in rows) == {p: 24 for p in {r['project'] for r in rows}}
        identities = {r['id']: case_identity(r) for r in rows}
        if reference is None:
            reference = identities
        else:
            assert identities == reference, f'Changed data identity for {method}/{seed}'
        tau = lock['thresholds'][method]['threshold']
        tie_cases = Counter()
        for r in rows:
            assert r['split'] == 'test' and r['label'] in (0, 1)
            source_case = test_cases[r['id']]
            for key in ('project', 'cluster_id', 'base_id', 'split', 'label'):
                assert r[key] == source_case[key]
            assert set(r['target_paths']) == set(source_case['target_paths'])
            expected_ids = {f"{uid}::{f['path']}" for uid in source_case['unit_ids'] for f in units[uid]['files']}
            assert {f['id'] for f in r['files']} == expected_ids
            paths = [f['path'] for f in r['files']]
            assert len(paths) == len(set(paths))
            targets = set(r['target_paths'])
            assert targets <= set(paths) and bool(targets) == bool(r['label'])
            assert r['score'] == max(f['score'] for f in r['files'])
            for f in r['files']:
                assert math.isfinite(f['score'])
                source = source_files[f['id']]
                assert source['path'] == f['path']
                assert source['changed_lines'] == f['changed_lines']
                # Corpus patches are hunk-only: additions/deletions, no file headers.
                patch_cost = sum(line.startswith(('+', '-')) for line in source['patch'].splitlines())
                assert patch_cost == f['changed_lines'] > 0
                unique_test_files.add(f['id'])
            for policy in POLICIES:
                ranked = ordered_files(r['files'], tau, policy)
                tie_cases[policy] += int(len({f['score'] for f in ranked}) < len(ranked))
                for budget in BUDGETS:
                    inspected, used = inspect_prefix(ranked, budget)
                    independently_verify_prefix(ranked, budget, inspected, used)
                    hits = len(targets & {f['path'] for f in inspected})
                    all_case_rows.append({
                        'method': method, 'run': str(seed), 'policy': policy, 'budget_changed_lines': budget,
                        'case_id': r['id'], 'project': r['project'], 'cluster_id': r['cluster_id'],
                        'label': r['label'], 'threshold': tau, 'eligible_files': len(r['files']),
                        'eligible_changed_lines': sum(f['changed_lines'] for f in r['files']),
                        'candidate_files': len(ranked), 'target_files': len(targets),
                        'inspected_files': len(inspected), 'inspected_changed_lines': used,
                        'unused_budget_changed_lines': budget - used, 'target_hits': hits,
                        'target_recall': hits / len(targets) if targets else None,
                        'target_any_hit': int(hits > 0) if targets else None,
                        'inspected_file_ids_json': json.dumps([f['id'] for f in inspected]),
                    })
        run_metadata.append({'method': method, 'run': str(seed), 'threshold': tau,
                             'unique_cases': 96, 'unique_positive_cases': 48,
                             'cases_with_exact_score_ties': dict(tie_cases)})
    per_run = []
    projects = sorted({r['project'] for r in all_case_rows})
    for meta in run_metadata:
        for policy in POLICIES:
            for budget in BUDGETS:
                for project in ['ALL'] + projects:
                    rows = [r for r in all_case_rows if r['method'] == meta['method'] and r['run'] == meta['run']
                            and r['policy'] == policy and r['budget_changed_lines'] == budget
                            and (project == 'ALL' or r['project'] == project)]
                    positive = [r for r in rows if r['label'] == 1]
                    per_run.append({'method': meta['method'], 'run': meta['run'], 'policy': policy,
                                    'budget_changed_lines': budget, 'project': project,
                                    'n_all_cases': len(rows), 'n_positive_cases': len(positive),
                                    'macro_target_recall': mean(r['target_recall'] for r in positive),
                                    'target_any_hit_fraction': mean(r['target_any_hit'] for r in positive),
                                    'mean_inspected_lines_all_cases': mean(r['inspected_changed_lines'] for r in rows),
                                    'mean_inspected_lines_positive_cases': mean(r['inspected_changed_lines'] for r in positive),
                                    'mean_inspected_files_all_cases': mean(r['inspected_files'] for r in rows),
                                    'mean_inspected_files_positive_cases': mean(r['inspected_files'] for r in positive),
                                    'mean_unused_budget_all_cases': mean(r['unused_budget_changed_lines'] for r in rows),
                                    'zero_inspection_fraction_all_cases': mean(r['inspected_files'] == 0 for r in rows),
                                    'zero_inspection_fraction_positive_cases': mean(r['inspected_files'] == 0 for r in positive)})
    summary = []
    metrics = list(per_run[0])[7:]
    for method in METHODS:
        for policy in POLICIES:
            for budget in BUDGETS:
                for project in ['ALL'] + projects:
                    rows = [r for r in per_run if r['method'] == method and r['policy'] == policy
                            and r['budget_changed_lines'] == budget and r['project'] == project]
                    entry = {'method': method, 'policy': policy, 'budget_changed_lines': budget,
                             'project': project, 'n_runs': len(rows), 'n_unique_cases': rows[0]['n_all_cases'],
                             'n_unique_positive_cases': rows[0]['n_positive_cases']}
                    for metric in metrics:
                        values = [r[metric] for r in rows]
                        entry[metric + '_mean'] = mean(values)
                        entry[metric + '_run_sd'] = stdev(values) if len(values) > 1 else None
                    summary.append(entry)
    assert len(unique_test_files) == 268
    assert len(all_case_rows) == 9 * 2 * 6 * 96
    assert len(per_run) == 9 * 2 * 6 * 5
    assert len(summary) == 7 * 2 * 6 * 5
    for group in summary:
        assert 0 <= group['macro_target_recall_mean'] <= 1
        assert 0 <= group['mean_inspected_lines_all_cases_mean'] <= group['budget_changed_lines']
    # Ensure post-analysis processing did not alter the frozen input/protocol bytes.
    assert all(sha(root / p) == input_hashes[p.as_posix()] for p in input_paths)
    assert sha(protocol_path) == receipt['protocol_sha256']
    dump_csv(out / 'line_budget_cases.csv', all_case_rows)
    dump_csv(out / 'line_budget_per_run.csv', per_run)
    dump_csv(out / 'line_budget_summary.csv', summary)
    payload = {'protocol': protocol, 'input_receipt': receipt, 'run_metadata': run_metadata,
               'verified_unique_test_files': len(unique_test_files), 'case_budget_observations': len(all_case_rows),
               'self_checks': tests, 'summary': summary,
               'completed_utc': datetime.now(timezone.utc).isoformat()}
    save(out / 'LINE_BUDGET_RESULTS.json', payload)
    save(out / 'LINE_BUDGET_VERIFICATION.json', {
        'all_input_hashes_unchanged': True, 'protocol_hash_unchanged': True,
        'same_cases_targets_and_eligible_files_for_all_runs': True,
        'patch_costs_recounted_for_all_268_unique_test_files': True,
        'independent_cumulative_cost_check_for_every_policy_case_and_budget': True,
        'checked_case_budget_rows': len(all_case_rows), 'synthetic_boundary_checks': tests,
        'statistical_scope': 'descriptive only; repeated seed SD is not a confidence interval',
        'outcomes_used_to_change_protocol': False})
    print(json.dumps({'out': str(out), 'case_budget_rows': len(all_case_rows), 'summary_rows': len(summary),
                      'protocol_sha256': receipt['protocol_sha256']}, indent=2))


if __name__ == '__main__':
    main()
