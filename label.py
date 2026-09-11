"""Batch command line interface; GUI users can run app.py instead."""
from pathlib import Path
import argparse,json,time
import joblib
from engine import read_cloud,infer,save_cloud

def main():
 p=argparse.ArgumentParser(description='PLY 자동 라벨링 (원본 보존)');p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',type=float,default=None);p.add_argument('--model',type=Path,default=Path(__file__).parent/'model.joblib');args=p.parse_args()
 model=joblib.load(args.model);files=sorted(args.input.glob('*.ply')) if args.input.is_dir() else [args.input]
 if not files:p.error('입력 PLY 파일이 없습니다.')
 if args.split is not None and not 0<args.split<1:p.error('--split은 0과 1 사이여야 합니다.')
 args.output.mkdir(parents=True,exist_ok=True);rows=[]
 for path in files:
  try:
   ply,_=read_cloud(path);start=time.monotonic();labels,review,_,_,info=infer(ply,model,split=args.split);dest=args.output/f'{path.stem}_labeled.ply';save_cloud(ply,labels,dest,review);rows.append(dict(file=str(path),output=str(dest),seconds=round(time.monotonic()-start,3),**info));print(rows[-1],flush=True)
  except Exception as e:rows.append(dict(file=str(path),error=str(e)));print(rows[-1],flush=True)
 (args.output/f'run_{time.time_ns()}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 if any('error' in row for row in rows):raise SystemExit(1)
if __name__=='__main__':main()
