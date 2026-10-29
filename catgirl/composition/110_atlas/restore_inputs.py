"""Restore exact key predictor packets/runtime job; never overwrite changed files."""
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent

with zipfile.ZipFile(ROOT/'reproduction_inputs.zip') as archive:
    for name in archive.namelist():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert (path.parts[0]=='packets.local' and path.suffix=='.json') or name=='runtime_job.local.json'
        target=ROOT/path
        data=archive.read(name)
        if target.exists():assert target.read_bytes()==data, f'Existing input changed: {name}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        print(name,hashlib.sha256(data).hexdigest())
