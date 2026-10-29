"""Build or verify a self-contained source/data archive; no model weights.

Defaults to study files plus the small historical inference dependencies. Extra
review files outside the study can be included explicitly. No network is used.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STUDY_PREFIX = PurePosixPath('catgirl/composition/114_paws_genki')
EXTENSIONS = {'.json', '.jsonl', '.csv', '.md', '.py', '.txt', '.html', '.png', '.svg'}
EXCLUDE_PARTS = {'__pycache__', '.git', '.venv', 'model-cache', 'agent_transcripts', 'transcripts'}
DEPENDENCIES = ['catgirl/runner.py', 'catgirl/layer_schedules.py', 'catgirl/public_control_runner.py',
                'catgirl/composition_worker.py', 'catgirl/public_qwen35_norms.json',
                'catgirl/composition/113_catgirl_circuits/run_circuits.py',
                'catgirl/composition/113_catgirl_circuits/MODIFIER_SOURCES.json',
                'catgirl/composition/113_catgirl_circuits/numeric_results/labels.csv',
                'catgirl/composition/113_catgirl_circuits/RESULTS.md']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def acceptable(path):
    return (path.suffix.lower() in EXTENSIONS and not set(path.parts) & EXCLUDE_PARTS
            and 'transcript' not in path.name.lower() and not path.is_symlink())


def collect(study=HERE, project=PROJECT, extras=()):
    entries = {}
    for path in sorted(study.rglob('*')):
        if path.is_file() and not path.is_symlink() and acceptable(path.relative_to(study)):
            entries[str(STUDY_PREFIX / path.relative_to(study).as_posix())] = path
    for name in DEPENDENCIES:
        path = project / name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing inference dependency: ' + name)
        entries[name] = path
    for index, supplied in enumerate(extras):
        supplied = Path(supplied).resolve()
        if supplied.is_dir():
            for path in sorted(supplied.rglob('*')):
                if path.is_file() and not path.is_symlink() and acceptable(path.relative_to(supplied)):
                    name = str(PurePosixPath('extra') / f'{index:02d}_{supplied.name}' / path.relative_to(supplied).as_posix())
                    entries[name] = path
        elif supplied.is_file() and acceptable(supplied):
            entries[str(PurePosixPath('extra') / f'{index:02d}_{supplied.name}')] = supplied
        else:
            raise ValueError('Extra input missing or excluded: ' + str(supplied))
    return entries


def build(output, extras=(), study=HERE, project=PROJECT):
    if output.exists():
        raise ValueError('Refusing to overwrite an existing recovery archive')
    entries = collect(study, project, extras)
    blobs = {name: path.read_bytes() for name, path in entries.items()}
    manifest = dict(format='study114-recovery-v1', files={
        name: dict(sha256=digest(data), bytes=len(data)) for name, data in sorted(blobs.items())},
        excluded='Model weights, caches, logs, full agent transcripts, and other archives.',
        rebuild='See catgirl/composition/114_paws_genki/RECOVERY.md after extraction.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(blobs.items()):
            archive.writestr(name, data)
        archive.writestr('RECOVERY_MANIFEST.json', json.dumps(manifest, indent=2))
    verified = verify(output)
    return dict(archive=str(output), archive_sha256=digest(output.read_bytes()),
                compressed_bytes=output.stat().st_size, **verified)


def verify(path, extract=None):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name:
                raise ValueError('Unsafe archive member')
        manifest = json.loads(archive.read('RECOVERY_MANIFEST.json'))
        if manifest['format'] != 'study114-recovery-v1':
            raise ValueError('Unknown archive format')
        if set(names) != set(manifest['files']) | {'RECOVERY_MANIFEST.json'}:
            raise ValueError('Manifest membership mismatch')
        total = 0
        for name, metadata in manifest['files'].items():
            data = archive.read(name)
            if digest(data) != metadata['sha256'] or len(data) != metadata['bytes']:
                raise ValueError('Corrupt archive member: ' + name)
            total += len(data)
        if extract:
            extract = Path(extract)
            if extract.exists() and (not extract.is_dir() or any(extract.iterdir())):
                raise ValueError('Extraction destination must be absent or empty')
            extract.mkdir(parents=True, exist_ok=True)
            for name in names:
                target = extract / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(archive.read(name))
    return dict(verified=True, files=len(manifest['files']), uncompressed_bytes=total)


def selftest():
    with tempfile.TemporaryDirectory(prefix='study114-recovery-') as temp:
        root = Path(temp)
        project = root / 'project'
        study = project / 'catgirl/composition/114_paws_genki'
        study.mkdir(parents=True)
        (study / 'example.jsonl').write_text('{"text":"nya","intervention_trace_summary":[]}\n')
        (study / 'agent_transcript.json').write_text('excluded')
        (study / 'weights.safetensors').write_bytes(b'excluded')
        for name in DEPENDENCIES:
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('test dependency')
        archive = root / 'recovery.zip'
        result = build(archive, study=study, project=project)
        assert result['files'] == len(DEPENDENCIES) + 1
        verify(archive, root / 'restored')
        assert (root / 'restored/catgirl/composition/114_paws_genki/example.jsonl').read_bytes() == (study / 'example.jsonl').read_bytes()
        try:
            build(archive, study=study, project=project)
            raise AssertionError('Overwrite allowed')
        except ValueError:
            pass
    print('Recovery synthetic test passed: exact bytes, exclusion, integrity, extraction, overwrite guard.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--extra', type=Path, action='append', default=[])
    p.add_argument('--verify', type=Path)
    p.add_argument('--extract', type=Path)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.verify:
        print(json.dumps(verify(a.verify, a.extract)))
    elif a.output:
        if a.extract:
            p.error('--extract requires --verify')
        print(json.dumps(build(a.output, a.extra)))
    else:
        p.error('Choose --output, --verify, or --selftest')


if __name__ == '__main__':
    main()
