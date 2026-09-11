"""Compare models on the twelve evaluation files retained from v2."""
import json
import joblib
import numpy as np
from train_v3 import ROOT, OUT, METRICS, save
from engine import read_cloud, truth_on_raw, infer, evaluate

def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    models = {'v2': joblib.load(ROOT/'model.joblib'), 'v3': joblib.load(OUT/'model_v3.joblib')}
    models['v2']['default_split'] = .06
    records = []
    for row in manifest['files']:
        if row['split'] not in ('old_test', 'new_test'):
            continue
        ply, _ = read_cloud(row['raw'])
        truth = truth_on_raw(ply, row['truth'])
        for name, model in models.items():
            labels, *_ = infer(ply, model)
            records.append(dict(id=row['id'], group=row['split'], model=name, **evaluate(truth, labels)))
        print('RETAINED TEST', row['id'], flush=True)
    summary = {group: {name: {metric: float(np.mean([r[metric] for r in records if r['group']==group and r['model']==name])) for metric in METRICS} for name in models} for group in ('old_test','new_test')}
    save('retained_evaluation.json', dict(summary=summary, records=records))
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
