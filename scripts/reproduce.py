"""Verify/extract saved data and recompute article calculations without inference."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
from download_artifacts import extract_checked,digest,filesystem_path

ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--archives-dir',type=Path,default=ROOT/'artifacts/archives')
 parser.add_argument('--work-dir',type=Path,default=ROOT/'work')
 parser.add_argument('--out-dir',type=Path,default=ROOT/'replay')
 args=parser.parse_args()
 args.archives_dir=filesystem_path(args.archives_dir)
 args.work_dir=filesystem_path(args.work_dir)
 args.out_dir=filesystem_path(args.out_dir)
 if args.out_dir.exists():raise SystemExit('Output directory already exists; choose a new --out-dir.')
 if args.work_dir.exists():raise SystemExit('Work directory already exists; choose a new --work-dir.')
 expected={n:h for h,n in (line.split(maxsplit=1) for line in (ROOT/'ARCHIVES_SHA256SUMS.txt').read_text().splitlines())}
 layout={'VTSE_2711_REPLICATION.zip':'primary','VTSE_2711_EXTERNAL_VALIDATION.zip':'external','VTSE_2711_NATURAL_EXPANSION.zip':'natural','VTSE_2711_REASSESSMENT.zip':'reassessment'}
 for name,folder in layout.items():
  archive=args.archives_dir/name
  if not archive.exists():raise SystemExit('Missing data asset: '+str(archive))
  if digest(archive)!=expected[name]:raise SystemExit('Archive SHA-256 mismatch: '+name)
 args.work_dir.mkdir(parents=True);args.out_dir.mkdir(parents=True)
 for name,folder in layout.items():extract_checked(args.archives_dir/name,args.work_dir/folder)
 primary=args.work_dir/'primary/replication'
 natural=args.work_dir/'natural/natural_expansion'
 external=args.work_dir/'external/replication/external_gold_validation_20260930'
 external_runner=args.work_dir/'external_runner'
 shutil.copytree(ROOT/'code/external',external_runner)
 jobs=[
  [ROOT/'code/audit_primary.py','--research-root',primary,'--output',args.out_dir/'independent'],
  [ROOT/'code/audit_bootstrap.py','--research-root',primary,'--output',args.out_dir/'independent'],
  [ROOT/'code/audit_natural.py','--natural-root',natural,'--output',args.out_dir/'independent'],
  [external_runner/'audit_external.py',external],
  [external_runner/'diagnose_context.py',external],
  [external_runner/'make_audit_report.py',external],
  [ROOT/'code/analyze_line_budgets.py','--root',primary,'--out',args.out_dir/'line_budgets']]
 receipt=[]
 for i,job in enumerate(jobs,1):
  print(f'[{i}/{len(jobs)}] {Path(job[0]).name}',flush=True)
  log=args.out_dir/f'{i:02d}_{Path(job[0]).stem}.log'
  with log.open('w',encoding='utf8') as out:
   result=subprocess.run([sys.executable,str(ROOT/'scripts/run_offline.py'),*map(str,job)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
  receipt.append({'script':Path(job[0]).name,'exit_code':result.returncode,'log':log.name})
  if result.returncode:
   (args.out_dir/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
   raise SystemExit('Replay failed; inspect '+str(log))
 shutil.copytree(external_runner,args.out_dir/'external',ignore=shutil.ignore_patterns('*.py','inputs','__pycache__'))
 (args.out_dir/'REPLAY_RECEIPT.json').write_text(json.dumps({'all_checks_passed':True,'jobs':receipt,'scope':'Saved observations only; not new inference or human annotation.'},indent=2),encoding='utf8')
 print('PASS: saved-output calculations reproduced. Results:',args.out_dir.resolve())

if __name__=='__main__':main()
