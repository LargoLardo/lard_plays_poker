"""Serve the existing frontend with selectable local training checkpoints."""

import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
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
                if magic not in (b'LARDCPP1', b'LARDCPP2', b'LARDCPP3', b'LARDCPP4', b'LARDCPP5'):
                    continue
                source.seek(24 if magic in (b'LARDCPP4', b'LARDCPP5') else 16)
                row = source.read(8)
                if len(row) != 8:
                    continue
                iterations, = struct.unpack('<Q', row)
                row = source.read(8)
                if len(row) != 8:
                    continue
                nodes, = struct.unpack('<Q', row)
        relative = path.relative_to(root.resolve()).as_posix()
        label = relative.removesuffix(path.suffix)
        if iterations is not None:
            label = f'{iterations / 1_000_000:g}M · {label}'
        key = checkpoint_id(path, root.resolve(), path.stat())
        records.append(dict(id=hashlib.sha256(relative.encode()).hexdigest()[:16], version=key,
                            label=label, source=path, iterations=iterations,
                            native=path.suffix == '.bin' and magic == b'LARDCPP5',
                            totalNodes=nodes if path.suffix == '.bin' else None,
                            preflop=f'/api/nodesets/{key}/preflop-model.json',
                            postflop=f'/api/nodesets/{key}/postflop-model.json'))
    return sorted(records, key=lambda item: (-(item['iterations'] or 0), item['label']))


class NodesetServer(ThreadingHTTPServer):
    def __init__(self, address, *, nodesets=ROOT / 'nodesets', cache=ROOT / 'artifacts/web-models'):
        self.nodesets, self.cache = Path(nodesets).resolve(), Path(cache).resolve()
        self.export_lock = threading.Lock()
        self.arena_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.inference_process = None
        self.inference_version = None
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
                    command = [str(build(model='v5' if entry['native'] else 'v4')), '--resume', str(copied), '--iterations', '0',
                               '--memory-mb', str(memory_mb),
                               '--output', str(temporary / 'export-copy.bin'), '--export', str(exported)]
                    if entry['native']:
                        command.extend(['--cache-mb', '0', '--max-nodes', '0'])
                else:
                    venv_python = ROOT / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
                    python = str(venv_python) if venv_python.exists() else sys.executable
                    command = [python, str(ROOT / 'tools/export_web_model.py'), str(copied),
                               str(exported / 'preflop-model.json'), '--postflop-output',
                               str(exported / 'postflop-model.json')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                exported.rename(target)
            return target

    def stop_inference(self):
        if self.inference_process is not None:
            process = self.inference_process
            self.inference_process = None
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout:
                process.stdout.close()
            self.inference_version = None

    def decision(self, key, line):
        with self.inference_lock:
            entry = next((item for item in checkpoints(self.nodesets) if item['id'] == key and item['native']), None)
            if entry is None:
                raise ValueError('Choose an available V5 checkpoint')
            if self.inference_version != entry['version'] or self.inference_process is None or self.inference_process.poll() is not None:
                self.stop_inference()
                memory_mb = max(256, (entry['source'].stat().st_size * 8 + 1048575) // 1048576)
                self.inference_process = subprocess.Popen(
                    [str(build(model='v5')), '--resume', str(entry['source']), '--infer', '--memory-mb', str(memory_mb), '--cache-mb', '32', '--max-nodes', '0'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
                self.inference_version = entry['version']
            process = self.inference_process
            try:
                process.stdin.write(line + '\n')
                process.stdin.flush()
                output = process.stdout.readline()
                if not output:
                    raise RuntimeError('Native inference process stopped')
                result = json.loads(output)
                if 'error' in result:
                    raise ValueError(result['error'])
                return result
            except (OSError, RuntimeError):
                self.stop_inference()
                raise

    def server_close(self):
        super().server_close()
        with self.inference_lock:
            self.stop_inference()


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

    def do_POST(self):
        if urlsplit(self.path).path == '/api/decision':
            self.do_decision()
            return
        if urlsplit(self.path).path != '/api/arena':
            self.send_error(404)
            return
        locked = False
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid arena request size')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('Invalid arena request')
            hands, seed = request.get('hands', 10_000), request.get('seed', 1)
            if type(hands) is not int or not 2 <= hands <= 1_000_000 or hands % 2:
                raise ValueError('Choose an even hand count between 2 and 1,000,000')
            if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                raise ValueError('Seed must be 0..4294967295')
            sources = {item['id']: item['source'] for item in checkpoints(self.server.nodesets)}
            sources['bundled'] = ROOT / 'FULLGAME_10m_iters.pkl'
            sources.update({f'baseline:{name}': f'baseline:{name}' for name in ('random', 'call', 'pot')})
            selected = [sources.get(request.get(label)) for label in ('a', 'b')]
            if any(path is None for path in selected):
                raise ValueError('Checkpoint was removed; refresh the page and choose again')
            locked = self.server.arena_lock.acquire(blocking=False)
            if not locked:
                self.send_bytes(json.dumps({'error': 'An arena match is already running.'}).encode(), 'application/json', 409)
                return
            venv_python = ROOT / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            python = str(venv_python) if venv_python.exists() else sys.executable
            command = [python, str(ROOT / 'agent_arena.py'), *map(str, selected),
                       '--hands', str(hands), '--seed', str(seed)]
            for label, path in zip(('a', 'b'), selected):
                if path == ROOT / 'FULLGAME_10m_iters.pkl':
                    command.append(f'--swap-{label}-legacy-positions')
            result = subprocess.run(command, check=True, capture_output=True)
            self.send_bytes(result.stdout, 'application/json')
        except (ValueError, TypeError) as error:
            self.send_bytes(json.dumps({'error': str(error)}).encode(), 'application/json', 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, subprocess.CalledProcessError) as error:
            print(f'Arena failed: {error}', file=sys.stderr)
            self.send_bytes(json.dumps({'error': 'Arena failed. Check the server terminal.'}).encode(), 'application/json', 500)
        finally:
            if locked:
                self.server.arena_lock.release()

    def do_decision(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('Invalid decision request size')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('Invalid decision request')
            hero, board, history = request.get('hero'), request.get('board'), request.get('history')
            actor, street = request.get('actor'), request.get('street')
            if not isinstance(hero, list) or len(hero) != 2 or not isinstance(board, list) or len(board) not in (0, 3, 4, 5):
                raise ValueError('Invalid visible cards')
            cards = hero + board
            if any(not isinstance(card, str) or len(card) != 2 or card[0] not in '23456789TJQKA' or card[1] not in 'shdc' for card in cards) or len(set(cards)) != len(cards):
                raise ValueError('Invalid or duplicate cards')
            if type(actor) is not int or actor not in (0, 1) or type(street) is not int or street not in (0, 1, 2, 3):
                raise ValueError('Invalid actor or street')
            if len(board) != (0 if street == 0 else street + 2) or not isinstance(history, list) or len(history) > 40:
                raise ValueError('Invalid board or history length')
            tokens = [*hero, str(len(board)), *board, str(len(history))]
            for event in history:
                if not isinstance(event, list) or len(event) != 3:
                    raise ValueError('Invalid observed action')
                kind, ratio, jam = event
                if type(kind) is not int or kind not in (0, 1, 2) or type(ratio) not in (float, int) or not math.isfinite(ratio) or not 0 <= ratio <= 1000 or type(jam) is not int or jam not in (0, 1):
                    raise ValueError('Invalid observed action')
                tokens.extend(map(str, event))
            tokens.extend((str(actor), str(street)))
            result = self.server.decision(request.get('nodeset'), ' '.join(tokens))
            self.send_bytes(json.dumps(result).encode(), 'application/json')
        except (ValueError, TypeError) as error:
            self.send_bytes(json.dumps({'error': str(error)}).encode(), 'application/json', 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f'Native decision failed: {error}', file=sys.stderr)
            self.send_bytes(json.dumps({'error': 'Native decision failed. Check the server terminal.'}).encode(), 'application/json', 500)

    def send_bytes(self, contents, content_type, status=200):
        self.send_response(status)
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
