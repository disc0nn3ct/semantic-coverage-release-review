"""Fetch the fixed v1.0.0 data asset, or verify/extract a local offline copy."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,shutil,urllib.request,zipfile,os

ROOT=Path(__file__).resolve().parents[1]
NAME='VTSE_2711_PUBLIC_DATA_v1.0.0.zip'
URL='https://github.com/disc0nn3ct/semantic-coverage-release-review/releases/download/v1.0.0/'+NAME

def filesystem_path(path):
 """Use Windows extended paths for deep archived upstream source trees."""
 resolved=str(Path(path).resolve())
 if os.name=='nt' and not resolved.startswith('\\\\?\\'):
  resolved='\\\\?\\UNC\\'+resolved[2:] if resolved.startswith('\\\\') else '\\\\?\\'+resolved
 return Path(resolved)

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def extract_checked(archive,destination):
 destination=filesystem_path(destination);destination.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP paths')
  for name in names:
   normalized=PurePosixPath(name.replace('\\','/'))
   if normalized.is_absolute() or '..' in normalized.parts or ':' in name:
    raise ValueError('Unsafe archive path')
   target=(destination/str(normalized)).resolve()
   if not target.is_relative_to(destination):raise ValueError('ZIP path escapes destination')
  if z.testzip() is not None:raise ValueError('ZIP CRC failure')
  z.extractall(destination)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--file',type=Path,help='Existing release ZIP: no network used')
 parser.add_argument('--out',type=Path,default=ROOT/'artifacts')
 args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 expected={n:h for h,n in (line.split(maxsplit=1) for line in (ROOT/'RELEASE_ASSETS_SHA256SUMS.txt').read_text().splitlines())}
 path=args.file.resolve() if args.file else args.out/NAME
 if not args.file and not path.exists():
  partial=path.with_suffix('.part')
  request=urllib.request.Request(URL,headers={'User-Agent':'semantic-coverage-release-review/1.0.0'})
  with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as out:
   shutil.copyfileobj(response,out)
  if digest(partial)!=expected[NAME]:raise ValueError('Downloaded asset SHA-256 mismatch')
  partial.replace(path)
 if digest(path)!=expected[NAME]:raise ValueError('Asset SHA-256 mismatch')
 extract_checked(path,args.out)
 print('Verified and extracted release data to',args.out.resolve())

if __name__=='__main__':main()
