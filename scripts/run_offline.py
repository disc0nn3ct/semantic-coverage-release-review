"""Run a saved-data Python calculation with Python socket connections disabled."""
import runpy,socket,sys
from pathlib import Path

def deny(*args,**kwargs):
 raise RuntimeError('Network disabled during saved-output replay')

if len(sys.argv)<2:raise SystemExit('Usage: run_offline.py SCRIPT [ARG ...]')
socket.socket.connect=deny
socket.socket.connect_ex=deny
socket.create_connection=deny
socket.getaddrinfo=deny
script=str(Path(sys.argv[1]).resolve())
sys.path.insert(0,str(Path(script).parent))
sys.argv=[script,*sys.argv[2:]]
runpy.run_path(script,run_name='__main__')
