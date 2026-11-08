"""Reproduce synthetic split-allocation evidence without media or model access."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from public_results import metrics_equal
from vision_manifest import assign_group_splits
from vision_split_examples import named_examples, stress_examples, summarize

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'docs/results/group_split_before_after.json'
BASE_COMMIT = '2beeb66d422b0270b05967c9b91d23555d060261'
BASE_SOURCE_SHA256 = 'ff952628253a1991e45f6adcad128b88aca92ed66bde67785ee59379cff126f9'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def sources():
    return {name: source_hash(ROOT/name) for name in
            ('code/vision_manifest.py', 'code/vision_split_examples.py', 'code/check_group_split_results.py')}


def record(example, implementation):
    rows = implementation(example['rows'], seed=example['seed'], train_ratio=example['train_ratio'],
                          validation_ratio=example['validation_ratio'])
    ratios = {'train': example['train_ratio'], 'validation': example['validation_ratio'],
              'test': 1-example['train_ratio']-example['validation_ratio']}
    result = summarize(rows, ratios)
    result['group_assignments'] = {r['group_id']: r['split'] for r in rows}
    result['assignment_sha256'] = digest(result['group_assignments'])
    return result


def comparison(example, baseline):
    before, after = record(example, baseline), record(example, assign_group_splits)
    return {'id': example['id'], 'fixture_sha256': digest(example), 'before': before, 'after': after}


def validate_saved(recorded, example):
    assignments = recorded['group_assignments']
    if set(assignments) != {r['group_id'] for r in example['rows']}:
        raise ValueError('Saved group identities differ from the fixture')
    def replay(rows, **kwargs):
        return [{**r,'split':assignments[r['group_id']]} for r in rows]
    if not metrics_equal(recorded,record(example,replay)):
        raise ValueError('Saved split metrics differ from recorded group assignments')


def outcomes(records):
    counts = Counter()
    for item in records:
        a, b = item['before']['allocation_objective'], item['after']['allocation_objective']
        counts['tied' if math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10) else 'better' if b<a else 'worse'] += 1
    return {name: counts[name] for name in ('better','tied','worse')}


def capture(baseline_source):
    # Only load the exact, reviewed baseline source, before executing its module.
    if source_hash(baseline_source) != BASE_SOURCE_SHA256:
        raise ValueError('Baseline source is not the pinned published revision')
    spec = importlib.util.spec_from_file_location('split_baseline', baseline_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    named = [comparison(e,module.assign_group_splits) for e in named_examples()]
    stress = [comparison(e,module.assign_group_splits) for e in stress_examples()]
    return {'schema':'synthetic_group_split_comparison_v2', 'baseline_commit':BASE_COMMIT,
            'baseline_source_sha256_lf':BASE_SOURCE_SHA256, 'candidate_sources_sha256_lf':sources(),
            'named_cases':named, 'stress_cases':stress, 'stress_outcomes':outcomes(stress),
            'fixtures_used_during_development':True, 'independent_evaluation_used':False,
            'calibration_reserve_used':False, 'model_training_performed':False, 'model_promoted':False,
            'limitations':'Fixed exploratory allocation examples; no statistical performance estimate, global-optimum guarantee, or independent evaluation.'}


def check(path=REPORT):
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data['schema'] != 'synthetic_group_split_comparison_v2'
            or data['baseline_commit'] != BASE_COMMIT or data['baseline_source_sha256_lf'] != BASE_SOURCE_SHA256
            or data['candidate_sources_sha256_lf'] != sources()):
        raise ValueError('Split comparison source identity changed')
    for name in ('independent_evaluation_used','calibration_reserve_used','model_training_performed','model_promoted'):
        if data.get(name) is not False:
            raise ValueError('Split comparison exceeds synthetic development scope')
    for key, examples in (('named_cases',named_examples()),('stress_cases',stress_examples())):
        saved = data[key]
        if len(saved) != len(examples):
            raise ValueError('Split comparison case count changed')
        for item, example in zip(saved,examples):
            if (item['id'] != example['id'] or item['fixture_sha256'] != digest(example)
                    or not metrics_equal(item['after'],record(example,assign_group_splits))):
                raise ValueError('Split comparison differs from recomputed allocation: '+example['id'])
            validate_saved(item['before'],example)
            if item['before']['group_leakage'] is not False or item['after']['group_leakage'] is not False:
                raise ValueError('Group leakage in split evidence')
    if data['stress_outcomes'] != outcomes(data['stress_cases']):
        raise ValueError('Split comparison summary differs from saved cases')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',type=Path,help='Exact published baseline file for a full before/after reproduction')
    parser.add_argument('--out',type=Path,help='New JSON capture path; existing files are preserved')
    parser.add_argument('--check',action='store_true',help='Verify committed evidence against the current implementation')
    args=parser.parse_args()
    if args.baseline_source:
        if not args.out or args.check:
            parser.error('Use --baseline-source with --out and without --check')
        result=capture(args.baseline_source)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print('Captured five named and 100 exploratory synthetic split comparisons.')
    else:
        if args.out:
            parser.error('--out requires --baseline-source')
        data=check()
        print('Split evidence verified: 5 named cases, 100 exploratory cases; '+str(data['stress_outcomes'])+'.')


if __name__=='__main__':
    main()
