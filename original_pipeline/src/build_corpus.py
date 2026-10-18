"""Freeze metadata-selected, count-balanced real-PR documentation-substitution cases.

No model calls, model outputs, remote imports or repository-code execution. All
packing and split order is deterministic from the precommitted salt and metadata.
"""
import collections
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data/upstream_candidates'
SALT = '20260926-vtse'
REPOS = ('fastapi/fastapi', 'apache/airflow', 'home-assistant/core', 'keycloak/keycloak')
MAX_TRIALS = 512
MAX_BASE_LINES = 600


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def order_key(context, value):
    return digest(SALT + '|' + context + '|' + str(value))


def word_count(text):
    return len(re.findall(r'[A-Za-z][A-Za-z0-9_]*', text))


def norm_text(text):
    return ' '.join(re.findall(r'[a-z0-9_]+', text.lower()))


def clean_description(title, raw_body):
    """Versioned mechanical source cleaning, without paraphrasing or truncation."""
    body = re.sub(r'<!--.*?-->', '', raw_body or '', flags=re.S).replace('\r\n', '\n')
    lines, skipped_level = [], None
    blocked_heading = re.compile(r'(?:ai disclaimer|ai transcript|checklist|type of change|additional information|was generative ai|generative ai|related issues?|related pull requests?|contributor checklist)', re.I)
    for line in body.splitlines():
        heading = re.match(r'^\s{0,3}(#{1,6})\s+(.*)', line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            if skipped_level is not None and level <= skipped_level:
                skipped_level = None
            if blocked_heading.search(text):
                skipped_level = level
            if skipped_level is not None:
                continue
            if text.lower() in {'pull request', 'description', 'proposed change', 'summary', 'breaking change'}:
                continue
            line = text
        if skipped_level is not None:
            continue
        if re.match(r'^\s*[-*]\s*\[[ xX]\]', line):
            continue
        if re.match(r'^\s*\[[^\]]+\]:\s*https?://', line):
            continue
        if re.match(r'^\s*(?:[-*]\s*)?(?:closes?|fix(?:es)?|resolves?|related(?: to)?|discussion|this pr (?:fixes or closes issue|is related to issue)|link to .*pull request|generated-by)\s*:?\s*(?:#\d*|https?://|<!--|$)', line, re.I):
            continue
        if re.search(r'(?:AI text below|Read the .*Pull Request Guidelines|For fundamental code changes, an Airflow Improvement Proposal)', line, re.I):
            continue
        if re.match(r'^\s*```', line):
            continue
        if re.fullmatch(r'\s*</?(?:details|summary)>\s*', line):
            continue
        lines.append(line)
    body = '\n'.join(lines)
    # Keep link caption but never expose source URLs or PR/issue identifiers.
    body = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', body)
    body = re.sub(r'https?://[^\s<>\)\]]+', '', body)
    body = re.sub(r'(?<!\w)#\d+\b', '', body)
    body = re.sub(r'(?m)^\s*[-*_]{3,}\s*$', '', body)
    # Remove recognized HTML elements only. Do not treat Java generics such as
    # <T>, <String, Integer>, comparisons, or unknown angle-bracket text as tags.
    known_tags = r'details|summary|br|p|div|span|table|tr|td|th|thead|tbody|ul|ol|li|strong|em|b|i|a|img|blockquote|pre|code'
    attributes = r'''(?:\s+[A-Za-z_:][A-Za-z0-9_:\-]*(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s<>]+))?)*'''
    body = re.sub(r'</?(?:'+known_tags+r')'+attributes+r'\s*/?>', '', body)
    body = '\n'.join(line for line in body.splitlines() if norm_text(line) != norm_text(title))
    body = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', body).strip()
    title = re.sub(r'https?://\S+|(?<!\w)#\d+\b', '', title).strip()
    title = re.sub(r'^[^A-Za-z0-9`]+', '', title).strip()
    return title, body, (title + ('\n\n' + body if body else '')).strip()


def file_patches(raw, selected_paths):
    """Retain original runtime hunk text; remove git index SHA/header metadata."""
    chunks = re.split(r'(?=^diff --git )', raw, flags=re.M)
    files = []
    for chunk in chunks:
        match = re.match(r'diff --git a/(.*?) b/([^\n]+)', chunk)
        if not match or match.group(2) not in selected_paths:
            continue
        lines = chunk.splitlines()
        first = next((i for i, line in enumerate(lines) if line.startswith('@@')), None)
        if first is None:
            raise ValueError('Runtime source file without a hunk')
        patch = '\n'.join(lines[first:]) + '\n'
        additions = sum(line.startswith('+') for line in lines[first:])
        deletions = sum(line.startswith('-') for line in lines[first:])
        files.append({'path': match.group(2), 'patch': patch, 'changed_lines': additions + deletions,
                      'additions': additions, 'deletions': deletions,
                      'original_file_diff_sha256': digest(chunk)})
    assert set(f['path'] for f in files) == set(selected_paths)
    return sorted(files, key=lambda f: f['path'])


def patch_signature(files):
    # Numeric code constants and operation signs are preserved. Only hunk
    # coordinates, unchanged context and inconsequential spacing are removed.
    parts = []
    for f in sorted(files, key=lambda f: f['path']):
        changes = [' '.join(line.split()) for line in f['patch'].splitlines() if line.startswith(('+','-'))]
        parts.append(f['path'] + '\n' + '\n'.join(changes))
    return digest('\n\n'.join(parts))


def shingles(text):
    words = norm_text(text).split()
    return set(zip(words, words[1:], words[2:])) if len(words) >= 3 else {tuple(words)}


def sha_set(unit):
    return {unit['base_sha'], unit['head_sha'], unit['merge_sha']}


def load_pool():
    all_units, screening = [], []
    input_hashes = {}
    for repo in REPOS:
        metadata_sources = {}
        for page in sorted((SOURCE / repo.replace('/', '__') / 'api').glob('closed_prs_page_*.json')):
            if page.name.endswith(('.download.json','.error.json')):
                continue
            download = json.loads(page.with_suffix('.json.download.json').read_text(encoding='utf-8'))
            data = page.read_bytes()
            assert digest(data) == download['sha256']
            page_path = str(page.relative_to(ROOT)).replace('\\','/')
            for pr in json.loads(data):
                metadata_sources.setdefault(pr['number'], {'acquired_utc':download['acquired_utc'],
                                            'path':page_path,'sha256':download['sha256']})
        candidate_file = SOURCE / repo.replace('/', '__') / 'eligible.json'
        input_hashes[str(candidate_file.relative_to(ROOT)).replace('\\','/')] = digest(candidate_file.read_bytes())
        for record in json.loads(candidate_file.read_text(encoding='utf-8')):
            uid = repo.replace('/', '__') + '__pr_' + str(record['number'])
            title, body, documentation = clean_description(record['title'], record['body'])
            words, body_words = word_count(documentation), word_count(body)
            reasons = []
            if words < 8 or len(documentation) < 40:
                reasons.append('combined_clean_description_below_8_words_or_40_chars')
            path = ROOT / record['diff_path']
            raw = path.read_bytes()
            assert digest(raw) == record['diff_sha256'], uid
            files = file_patches(raw.decode('utf-8'), record['runtime_paths'])
            assert sum(f['changed_lines'] for f in files) == record['runtime_changed_lines'], uid
            unit = {'id': uid, 'project': repo, 'pr_number': record['number'], 'url': record['original_url'],
                    'base_sha': record['base_sha'], 'head_sha': record['head_sha'], 'merge_sha': record['merge_commit_sha'],
                    'merged_at': record['merged_at'], 'artifact_acquired_utc': record['acquired_utc'],
                    'metadata_acquired_utc': metadata_sources[record['number']]['acquired_utc'],
                    'metadata_source_path': metadata_sources[record['number']]['path'],
                    'metadata_source_sha256': metadata_sources[record['number']]['sha256'],
                    'documentation': documentation, 'clean_title': title, 'clean_body': body,
                    'body_words': body_words, 'documentation_words': words,
                    'title_only': body_words == 0, 'body_under10': body_words < 10,
                    'author_id': record.get('author_id'), 'author_login': record.get('author_login'),
                    'author_association_at_acquisition': record.get('author_association'),
                    'source_diff_path': record['diff_path'], 'source_diff_sha256': record['diff_sha256'],
                    'raw_description_path': str((path.parent/'description_raw.md').relative_to(ROOT)).replace('\\','/'),
                    'files': files, 'runtime_changed_lines': sum(f['changed_lines'] for f in files),
                    'normalized_patch_sha256': patch_signature(files), 'normalized_documentation_sha256': digest(norm_text(documentation)),
                    'description_origin': 'contributor-authored current description of a merged PR; not independently verified complete release documentation'}
            screening.append({'id':uid,'project':repo,'description_words':words,'body_words':body_words,
                              'title_only':unit['title_only'],'body_under10':unit['body_under10'],'eligible':not reasons,'reasons':reasons})
            if not reasons:
                all_units.append(unit)
    # Stable representative choice and deduplication precede any grouping/split.
    all_units.sort(key=lambda u: order_key('candidate-order',u['id']))
    kept, duplicates = [], []
    for unit in all_units:
        reason, other_id = None, None
        tokens = shingles(unit['documentation'])
        for prior in kept:
            if unit['normalized_patch_sha256'] == prior['normalized_patch_sha256']:
                reason, other_id = 'same_normalized_runtime_patch', prior['id']; break
            if unit['normalized_documentation_sha256'] == prior['normalized_documentation_sha256']:
                reason, other_id = 'same_normalized_description', prior['id']; break
            previous = shingles(prior['documentation'])
            similarity = len(tokens & previous) / len(tokens | previous) if tokens | previous else 1
            if similarity >= .90:
                reason, other_id = 'description_3word_shingle_jaccard_at_least_0.90', prior['id']; break
        if reason:
            duplicates.append({'excluded_id':unit['id'],'kept_id':other_id,'reason':reason})
        else:
            kept.append(unit)
    return kept, screening, duplicates, input_hashes


def pack_project(units, repo):
    ids = {u['id']:u for u in units}
    paths = {u['id']:{f['path'] for f in u['files']} for u in units}
    shas = {u['id']:sha_set(u) for u in units}
    lines = {u['id']:u['runtime_changed_lines'] for u in units}
    diagnostics = {'project': repo,'pool_count':len(units),'unique_runtime_paths':len(set().union(*paths.values())) if paths else 0,
                   'path_frequencies':dict(collections.Counter(p for v in paths.values() for p in v)), 'trials':[]}
    if len(units) < 48:
        return None, diagnostics
    for trial in range(MAX_TRIALS):
        available = sorted(ids, key=lambda uid:order_key(f'packing|{repo}|{trial}',uid))
        clusters = []
        for cluster_index in range(8):
            found = None
            for ta, tb in itertools.combinations(available, 2):
                if paths[ta] & paths[tb] or shas[ta] & shas[tb]:
                    continue
                target_paths = paths[ta] | paths[tb]
                target_shas = shas[ta] | shas[tb]
                rest = [u for u in available if u not in (ta,tb) and not paths[u]&target_paths and not shas[u]&target_shas]
                if len(rest) < 4:
                    continue
                for a, b in itertools.combinations(rest, 2):
                    if paths[a]&paths[b] or shas[a]&shas[b] or lines[ta]+lines[a]+lines[b]>MAX_BASE_LINES:
                        continue
                    used_shas = shas[a] | shas[b]
                    for c, d in itertools.combinations([u for u in rest if u not in (a,b) and not shas[u]&used_shas],2):
                        if paths[c]&paths[d] or shas[c]&shas[d] or lines[tb]+lines[c]+lines[d]>MAX_BASE_LINES:
                            continue
                        found = ([ta,a,b],[tb,c,d],ta,tb)
                        break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
            a,b,ta,tb = found
            picked = a+b
            used_shas = set().union(*(shas[u] for u in picked))
            cluster_id = repo.replace('/','__') + '__cluster_' + order_key('cluster-id','|'.join(sorted(picked)))[:12]
            clusters.append({'id':cluster_id,'project':repo,'unit_ids':sorted(picked),
                             'base_a_unit_ids':a,'base_b_unit_ids':b,'target_a_unit_id':ta,'target_b_unit_id':tb})
            available = [u for u in available if u not in picked and not shas[u]&used_shas]
        diagnostics['trials'].append({'trial':trial,'complete_clusters':len(clusters),'remaining_units':len(available)})
        if len(clusters)==8:
            diagnostics['selected_first_feasible_trial'] = trial
            return clusters, diagnostics
    return None, diagnostics


def csv_out(path, rows):
    if not rows:
        path.write_text('',encoding='utf-8')
        return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorized-preseal-cleaner-fix',action='store_true')
    args=parser.parse_args()
    units, screening, duplicates, inputs = load_pool()
    diagnostics, clusters = [], []
    rootdata = ROOT / 'data'
    csv_out(rootdata/'description_screening.csv',screening)
    csv_out(rootdata/'candidate_deduplication.csv',duplicates)
    (rootdata/'eligible_clean_units.json').write_text(json.dumps(units,ensure_ascii=False,indent=2),encoding='utf-8')
    for repo in REPOS:
        result, report = pack_project([u for u in units if u['project']==repo],repo)
        diagnostics.append(report)
        (rootdata/'packing_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'project':repo,'clean_pool':report['pool_count'],'clusters':len(result or []),
                          'trial':report.get('selected_first_feasible_trial')},ensure_ascii=True),flush=True)
        if result is None:
            raise RuntimeError('Insufficient feasible metadata-only packing for '+repo+'; no frozen corpus written')
        result.sort(key=lambda c:order_key('split',c['id']))
        for i,cluster in enumerate(result):
            cluster['split'] = 'development' if i<2 else 'test'
        clusters.extend(result)
    selected_ids = {uid for c in clusters for uid in c['unit_ids']}
    selected = sorted((u for u in units if u['id'] in selected_ids),key=lambda u:u['id'])
    by_id = {u['id']:u for u in selected}
    cases=[]
    for cluster in clusters:
        for side,other in [('a','b'),('b','a')]:
            base_id=cluster['id']+'__'+side
            own=sorted(cluster['base_'+side+'_unit_ids'],key=lambda u:order_key('unit-order|'+base_id,u))
            target=cluster['target_'+side+'_unit_id']
            other_target=cluster['target_'+other+'_unit_id']
            for label in [0,1]:
                docs=[other_target if label and u==target else u for u in own]
                docs.sort(key=lambda u:order_key('document-order|'+base_id,u))
                cases.append({'id':base_id+('__substitution' if label else '__control'),
                              'project':cluster['project'],'split':cluster['split'],'cluster_id':cluster['id'],
                              'base_id':base_id,'label':label,'condition':'description_substitution' if label else 'complete_source_descriptions',
                              'unit_ids':own,'document_unit_ids':docs,'target_unit_id':target if label else None,
                              'target_paths':sorted(f['path'] for f in by_id[target]['files']) if label else []})
    corpus={'study':'Controlled documentation-substitution benchmark from real merged PR artifacts',
            'schema_version':1,'selection_salt':SALT,'units':selected,'cases':cases,'clusters':clusters,
            'interpretation':'Artificial bundles and source-description availability/substitution labels. Not natural releases, natural undocumented behavior, independently validated completeness or code injections.'}
    verify(corpus)
    payload=json.dumps(corpus,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    target=rootdata/'corpus.json'
    if target.exists() and target.read_bytes()!=payload.encode('utf-8'):
        old_raw=target.read_bytes(); old=json.loads(old_raw)
        old_compare=json.loads(old_raw); new_compare=json.loads(payload)
        metadata_fields={'metadata_acquired_utc','artifact_acquired_utc','metadata_source_path','metadata_source_sha256'}
        for obj in (old_compare,new_compare):
            for unit in obj['units']:
                for field in metadata_fields: unit.pop(field,None)
        if (old_compare!=new_compare and not args.authorized_preseal_cleaner_fix) or list(ROOT.rglob('*EXECUTION_SEAL*')):
            raise RuntimeError('Refusing to replace an existing different frozen corpus')
        # Parent explicitly authorized this exact timestamp-field correction
        # before any benchmark call. Preserve the earlier bytes for the audit.
        archive=rootdata/'preseal_versions';archive.mkdir(exist_ok=True)
        (archive/('corpus_'+digest(old_raw)+'.json')).write_bytes(old_raw)
        previous_manifest=rootdata/'corpus_freeze_manifest.json'
        if previous_manifest.exists():
            (archive/('freeze_manifest_'+digest(old_raw)+'.json')).write_bytes(previous_manifest.read_bytes())
        correction={
            'old_corpus_sha256':digest(old_raw),'new_corpus_sha256':digest(payload),
            'reason':('Authorized pre-seal cleaner correction: preserve generics/comparisons and strip only recognized HTML markup' if args.authorized_preseal_cleaner_fix else
                      'Pre-seal provenance/serialization correction: separate API and diff timestamps and write canonical UTF-8 LF bytes'),
            'prediction_inputs_cases_labels_splits_unchanged':old_compare==new_compare,
            'case_assignments_unchanged':old['cases']==corpus['cases'],
            'authorization':'Parent instruction before any benchmark execution or EXECUTION_SEAL',
            'corrected_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        history_path=rootdata/'preseal_correction_history.json'
        history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
        previous_record=rootdata/'metadata_correction_record.json'
        if not history and previous_record.exists(): history.append(json.loads(previous_record.read_text(encoding='utf-8')))
        history.append(correction)
        history_path.write_text(json.dumps(history,indent=2),encoding='utf-8')
    target.write_bytes(payload.encode('utf-8'))
    (rootdata/'corpus.sha256').write_text(digest(payload)+'  corpus.json\n',encoding='utf-8')
    manifest={'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'corpus_sha256':digest(payload),
              'builder_sha256':digest(Path(__file__).read_bytes()),'selection_salt':SALT,'input_hashes':inputs,
              'counts':{'units':len(selected),'clusters':len(clusters),'cases':len(cases),
                        'development_cases':sum(c['split']=='development' for c in cases),'test_cases':sum(c['split']=='test' for c in cases)},
              'selected_documentation_quality':{repo:{'units':sum(u['project']==repo for u in selected),
                  'title_only':sum(u['title_only'] for u in selected if u['project']==repo),
                  'body_under10':sum(u['body_under10'] for u in selected if u['project']==repo)} for repo in REPOS},
              'verification':'All schema/count/PR/SHA/split/path/target/document-frequency assertions passed',
              'limitations':['No independent semantic-completeness or annotation validation','Historical PR text is observed at acquisition, not authenticated as exact merge-time prose',
                             'No model outcomes were read; packing uses only hashes, file paths, code size and provenance identities','Title-only and short-body cases are retained and flagged, not silently excluded after outcomes']}
    (rootdata/'corpus_freeze_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    csv_out(rootdata/'split_manifest.csv',[{k:c[k] for k in ['id','project','split','cluster_id','base_id','label','unit_ids','document_unit_ids','target_unit_id','target_paths']} for c in cases])
    csv_out(rootdata/'unit_manifest.csv',[{k:u[k] for k in ['id','project','pr_number','url','base_sha','head_sha','merge_sha','merged_at','source_diff_sha256','body_words','title_only','body_under10','runtime_changed_lines']} for u in selected])
    print(json.dumps(manifest,ensure_ascii=True,indent=2),flush=True)


def verify(corpus):
    units={u['id']:u for u in corpus['units']}
    assert len(units)==192 and len(corpus['cases'])==128 and len(corpus['clusters'])==32
    seen_shas=set();seen_prs=set();seen_patches=set()
    for u in units.values():
        assert (u['project'],u['pr_number']) not in seen_prs
        seen_prs.add((u['project'],u['pr_number']))
        assert not seen_shas & sha_set(u),u['id']
        seen_shas.update(sha_set(u))
        assert u['normalized_patch_sha256'] not in seen_patches
        seen_patches.add(u['normalized_patch_sha256'])
        assert word_count(u['documentation'])>=8 and len(u['documentation'])>=40
        assert not re.search(r'https?://|(?<!\w)#\d+\b',u['documentation'])
    case_docs=collections.defaultdict(collections.Counter)
    memberships={};base_labels=collections.defaultdict(set)
    for c in corpus['cases']:
        assert len(set(c['unit_ids']))==len(c['unit_ids'])==3
        assert len(set(c['document_unit_ids']))==len(c['document_unit_ids'])==3
        assert sum(units[u]['runtime_changed_lines'] for u in c['unit_ids'])<=MAX_BASE_LINES
        paths=[{f['path'] for f in units[u]['files']} for u in c['unit_ids']]
        assert not any(a&b for a,b in itertools.combinations(paths,2))
        for u in c['unit_ids']+c['document_unit_ids']:
            assert units[u]['project']==c['project']
            if u in memberships: assert memberships[u]==(c['split'],c['cluster_id'])
            memberships[u]=(c['split'],c['cluster_id'])
        for u in c['document_unit_ids']:case_docs[u][c['label']]+=1
        base_labels[c['base_id']].add(c['label'])
        if not c['label']:
            assert not c['target_paths'] and c['target_unit_id'] is None
            assert set(c['unit_ids'])==set(c['document_unit_ids'])
        else:
            assert c['target_unit_id'] in c['unit_ids'] and c['target_unit_id'] not in c['document_unit_ids']
            assert len(set(c['unit_ids'])&set(c['document_unit_ids']))==2
            assert set(c['target_paths'])=={f['path'] for f in units[c['target_unit_id']]['files']}
    assert all(v=={0,1} for v in base_labels.values())
    assert all(v=={0:1,1:1} for v in case_docs.values())
    for cluster in corpus['clusters']:
        assert len(set(cluster['unit_ids']))==6
        for target in (cluster['target_a_unit_id'],cluster['target_b_unit_id']):
            tp={f['path'] for f in units[target]['files']}
            assert not any(tp & {f['path'] for f in units[u]['files']} for u in cluster['unit_ids'] if u!=target)
    for repo in REPOS:
        assert sum(u['project']==repo for u in units.values())==48
        assert sum(c['project']==repo and c['split']=='development' for c in corpus['cases'])==8
        assert sum(c['project']==repo and c['split']=='test' for c in corpus['cases'])==24


if __name__=='__main__':
    main()
