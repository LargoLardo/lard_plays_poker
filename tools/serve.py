"""Serve the existing frontend with selectable local training checkpoints."""

import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpp.run import build


def checkpoint_id(path, root, stat):
    stamp = f'{path.relative_to(root).as_posix()}:{stat.st_mtime_ns}:{stat.st_size}'
    return hashlib.sha256(stamp.encode()).hexdigest()[:24]


def checkpoints(root):
    records = []
    for path in sorted(root.rglob('*')):
        if path.suffix not in ('.bin', '.pkl') or not path.is_file():
            continue
        path = path.resolve()
        if not path.is_relative_to(root.resolve()):
            continue
        iterations = None
        if path.suffix == '.bin':
            with path.open('rb') as source:
                magic = source.read(8)
                if magic not in (b'LARDCPP1', b'LARDCPP2', b'LARDCPP3', b'LARDCPP4'):
                    continue
                source.seek(24 if magic == b'LARDCPP4' else 16)
                row = source.read(8)
                if len(row) != 8:
                    continue
                iterations, = struct.unpack('<Q', row)
        relative = path.relative_to(root.resolve()).as_posix()
        label = relative.removesuffix(path.suffix)
        if iterations is not None:
            label = f'{iterations / 1_000_000:g}M · {label}'
        key = checkpoint_id(path, root.resolve(), path.stat())
        records.append(dict(id=hashlib.sha256(relative.encode()).hexdigest()[:16], version=key,
                            label=label, source=path, iterations=iterations,
                            preflop=f'/api/nodesets/{key}/preflop-model.json',
                            postflop=f'/api/nodesets/{key}/postflop-model.json'))
    return sorted(records, key=lambda item: (-(item['iterations'] or 0), item['label']))


class NodesetServer(ThreadingHTTPServer):
    def __init__(self, address, *, nodesets=ROOT / 'nodesets', cache=ROOT / 'artifacts/web-models'):
        self.nodesets, self.cache = Path(nodesets).resolve(), Path(cache).resolve()
        self.export_lock = threading.Lock()
        super().__init__(address, Handler)

    def export(self, key):
        # Export only the selected model, once per saved checkpoint version.
        with self.export_lock:
            target = self.cache / key
            if (target / 'postflop-model.json').exists() and (target / 'preflop-model.json').exists():
                return target
            entry = next((item for item in checkpoints(self.nodesets) if item['version'] == key), None)
            if entry is None:
                raise FileNotFoundError('Checkpoint changed or was removed; refresh the page.')
            self.cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.cache) as directory:
                temporary = Path(directory)
                source = entry['source']
                copied = temporary / ('checkpoint' + source.suffix)
                with source.open('rb') as original, copied.open('wb') as output:
                    if checkpoint_id(source, self.nodesets, os.fstat(original.fileno())) != key:
                        raise FileNotFoundError('Checkpoint changed; refresh the page.')
                    shutil.copyfileobj(original, output)
                exported = temporary / 'web'
                if source.suffix == '.bin':
                    memory_mb = max(256, (copied.stat().st_size * 6 + 1_048_575) // 1_048_576)
                    command = [str(build()), '--resume', str(copied), '--iterations', '0',
                               '--memory-mb', str(memory_mb),
                               '--output', str(temporary / 'export-copy.bin'), '--export', str(exported)]
                else:
                    venv_python = ROOT / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
                    python = str(venv_python) if venv_python.exists() else sys.executable
                    command = [python, str(ROOT / 'tools/export_web_model.py'), str(copied),
                               str(exported / 'preflop-model.json'), '--postflop-output',
                               str(exported / 'postflop-model.json')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                exported.rename(target)
            return target


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'public'), **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == '/api/nodesets':
                items = [{k: v for k, v in item.items() if k != 'source'}
                         for item in checkpoints(self.server.nodesets)]
                self.send_bytes(json.dumps(items).encode(), 'application/json')
            elif path.startswith('/api/nodesets/'):
                parts = path.split('/')
                if (len(parts) != 5 or len(parts[3]) != 24
                        or any(c not in '0123456789abcdef' for c in parts[3])
                        or parts[4] not in ('preflop-model.json', 'postflop-model.json')):
                    self.send_error(404)
                    return
                exported = self.server.export(parts[3]) / parts[4]
                self.send_bytes(exported.read_bytes(), 'application/json')
            else:
                super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass  # A different selection cancels the browser's previous request.
        except FileNotFoundError as error:
            self.send_error(404, str(error))
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f'Nodeset export failed: {error}', file=sys.stderr)
            self.send_error(500, 'Could not export this nodeset. Check the server terminal.')

    def send_bytes(self, contents, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(contents)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(contents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    with NodesetServer(('127.0.0.1', args.port)) as server:
        print(f'Play and study at http://localhost:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
