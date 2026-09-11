"""Audit supplied pairs, train a candidate, and compare on held-out files."""
from pathlib import Path
import collections
import hashlib
import json
import subprocess
import sys
import numpy as np
import joblib
from engine import read_cloud, truth_on_raw, predict_grid, segment_prediction, evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'training_v3'
METRICS = ('pallet_iou', 'object_mean_iou', 'object_f1_at_50')

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = Path('/Users/hwangjaeyeon/Desktop/before')
    after = Path('/Users/hwangjaeyeon/Desktop/after')
    old = json.loads((ROOT / 'training_manifest_v2.json').read_text())
    previous = {Path(r['raw']).name: r['split'] for r in old['files']}
    files, excluded, seen = [], [], {}
    names = sorted(before.glob('*.ply'), key=lambda p: int(p.stem))
    for i, path in enumerate(names):
        try:
            truth = after / path.name
            raw, xyz = read_cloud(path)
            labeled, _ = read_cloud(truth)
            data = labeled['vertex'].data
            fields = [n for n in data.dtype.names if 'instance_label' in n]
            if fields != ['scalar_instance_label']:
                raise ValueError('Missing or ambiguous instance label: ' + str(fields))
            labels = data['scalar_instance_label']
            if not np.all(np.isfinite(labels) & (labels >= 1) & (labels == np.floor(labels))):
                raise ValueError('Invalid label values')
            if len(np.unique(labels)) < 2 or 1 not in labels:
                raise ValueError('Missing pallet or object class')
            y = truth_on_raw(raw, truth)
            fraction = float(np.mean(y > 0))
            if fraction < .95:
                raise ValueError('Coordinate coverage below 95%: ' + str(fraction))
            digest = hashlib.sha256(xyz.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError('Duplicate coordinates: ' + seen[digest])
            seen[digest] = path.name
            row = dict(id='pair_'+path.stem, raw=str(path), truth=str(truth),
                       label_source='scalar_instance_label', split=previous.get(path.name, 'pending'),
                       match_fraction=fraction, points=len(y), objects=int(len(np.unique(labels))-1),
                       coordinate_sha256=digest)
            files.append(row)
        except (ValueError, FileNotFoundError) as error:
            excluded.append(dict(file=path.name, reason=str(error)))
        if (i+1) % 20 == 0:
            print('AUDITED', i+1, len(names), flush=True)
    # Preserve every previous split; contiguous final blocks are new held-out data.
    fresh = [r for r in files if r['split'] == 'pending']
    for i, row in enumerate(fresh):
        row['split'] = 'test' if i >= len(fresh)-10 else 'validation' if i >= len(fresh)-20 else 'train'
    manifest = dict(files=files, excluded=excluded,
                    note='Previous splits preserved by filename. Final 10 new files test, preceding 10 validation. Related scans may remain correlated.')
    save('manifest.json', manifest)
    print('AUDIT COMPLETE', dict(collections.Counter(r['split'] for r in files)), 'excluded', len(excluded), flush=True)
    subprocess.run([sys.executable, str(ROOT/'retrain.py'), str(OUT/'manifest.json'),
                    '--cache', str(OUT/'cache'), '--output', str(OUT/'model_v3.joblib')], check=True)
    candidate = joblib.load(OUT/'model_v3.joblib')
    baseline = joblib.load(ROOT/'model.joblib')
    baseline['default_split'] = .06
    records = []
    thresholds = (.03, .06, .1, .15, .2)
    for row in files:
        if row['split'] != 'validation':
            continue
        ply, _ = read_cloud(row['raw']); y = truth_on_raw(ply, row['truth'])
        pred = predict_grid(ply, candidate)
        for threshold in thresholds:
            labels, *_ = segment_prediction(*pred, split=threshold)
            records.append(dict(id=row['id'], split=threshold, **evaluate(y, labels)))
        print('VALIDATED', row['id'], flush=True)
    best = max(thresholds, key=lambda t: np.mean([r['object_mean_iou'] for r in records if r['split']==t]))
    candidate['default_split'] = best
    joblib.dump(candidate, OUT/'model_v3.joblib', compress=3)
    save('validation.json', dict(selected_split=best, records=records))
    tests = []
    for row in files:
        if row['split'] != 'test':
            continue
        ply, _ = read_cloud(row['raw']); y = truth_on_raw(ply, row['truth'])
        for name, model in [('v2', baseline), ('v3', candidate)]:
            labels, *_ = segment_prediction(*predict_grid(ply, model), split=model['default_split'])
            tests.append(dict(id=row['id'], model=name, **evaluate(y, labels)))
        save('test_progress.json', tests)
        print('TESTED', row['id'], flush=True)
    summary = {name: {m: float(np.mean([r[m] for r in tests if r['model']==name])) for m in METRICS} for name in ('v2','v3')}
    save('evaluation.json', dict(summary=summary, selected_split=best, records=tests))
    print('COMPLETE', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
