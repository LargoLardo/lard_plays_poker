import hashlib
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from cpp.run import build
import pf_mccfr
from tools.serve import NodesetServer, checkpoints


class NodesetServerTests(unittest.TestCase):
    def test_default_catalog_includes_committed_models_without_duplicating_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            committed = root / 'checkpoints'
            committed.mkdir()
            (committed / 'v1.pkl').write_bytes(b'bundled legacy')
            for name, magic in (('v2', b'LARDCPP4'), ('v3', b'LARDCPP5')):
                (committed / (name + '.bin')).write_bytes(struct.pack('<8s4I2Q', magic, 0, 32, 1, 64, 500_000_000, 1))
            with patch('tools.serve.ROOT', root):
                with NodesetServer(('127.0.0.1', 0), nodesets=root / 'nodesets', cache=root / 'cache') as server:
                    entries = server.catalog()
                    self.assertEqual({item['label'] for item in entries}, {'V2 · 500M', 'V3 · 500M'})
                    self.assertTrue(all(item['sourceRoot'] == committed for item in entries))

    def test_v5_native_inference_lifecycle_and_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodesets, cache = root / 'nodesets', root / 'cache'
            nodesets.mkdir()
            binary = str(build(model='v5'))
            asset = root / 'cards.abs'
            subprocess.run([binary, '--build-abstraction', str(asset), '--examples', '32',
                            '--clusters', '4,4,4', '--samples', '32', '--cluster-rounds', '1'],
                           check=True, capture_output=True)
            native = nodesets / 'v5.bin'
            subprocess.run([binary, '--abstraction', str(asset), '--iterations', '100',
                            '--workers', '2', '--chunk-size', '4', '--output', str(native)],
                           check=True, capture_output=True)
            (nodesets / 'truncated.bin').write_bytes(b'LARDCPP5' + bytes(24))
            original = native.read_bytes()
            server = NodesetServer(('127.0.0.1', 0), nodesets=nodesets, cache=cache)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            address = f'http://127.0.0.1:{server.server_port}'

            def post(path, body):
                request = Request(address + path, method='POST', data=json.dumps(body).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=20) as response:
                    return json.load(response)

            try:
                with urlopen(address + '/api/nodesets') as response:
                    entries = json.load(response)
                self.assertEqual(len(entries), 1)
                entry = entries[0]
                self.assertTrue(entry['native'])
                self.assertTrue(entry['label'].startswith('V3 · '))
                self.assertNotIn('sourceRoot', entry)
                self.assertGreater(entry['totalNodes'], 0)
                with urlopen(address + entry['preflop'], timeout=20) as response:
                    rows = json.load(response)
                    self.assertTrue(rows)
                    self.assertTrue(all(len(row) == 7 for row in rows.values()))
                body = dict(nodeset=entry['id'], hero=['As', 'Kd'], board=[], history=[], actor=1, street=0)
                first = post('/api/decision', body)
                process = server.inference_process
                self.assertEqual(first, post('/api/decision', body))
                self.assertIs(process, server.inference_process, 'Keep the loaded native model')
                self.assertEqual(first['amounts'][2:], [2.5, 3, 100])
                body.update(history=[[1, 0, 0]], actor=0)
                self.assertFalse(post('/api/decision', body).get('unavailable', False))
                for changes in (dict(hero=['As', 'As']), dict(actor=2), dict(street=1),
                                dict(history=[[2, 'nan', 0]]), dict(nodeset='bundled')):
                    with self.assertRaises(HTTPError) as failed:
                        post('/api/decision', {**body, **changes})
                    self.assertEqual(failed.exception.code, 400)
                native.touch()
                post('/api/decision', body)
                self.assertIsNot(process, server.inference_process)
                self.assertIsNotNone(process.poll(), 'Release the previous checkpoint process')
                match = post('/api/arena', dict(a=entry['id'], b='baseline:random', hands=20, seed=11))
                self.assertEqual(match['b']['baseline'], 'random')
                current = server.inference_process
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
            self.assertIsNotNone(current.poll(), 'Closing the server releases native inference')
            self.assertEqual(native.read_bytes(), original)

    def test_checkpoint_discovery_lazy_export_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodesets, cache = root / 'nodesets', root / 'cache'
            nodesets.mkdir()
            native = nodesets / 'native.bin'
            python = nodesets / 'python.pkl'
            subprocess.run([str(build()), '--mode', 'preflop', '--iterations', '20',
                            '--samples', '3', '--output', str(native)], check=True, capture_output=True)
            pf_mccfr.train(20, samples=3, output=python)
            (nodesets / 'unfinished.bin.tmp').write_bytes(b'partial')
            (nodesets / 'unknown.bin').write_bytes(b'not a checkpoint')
            before = {path: hashlib.sha256(path.read_bytes()).digest() for path in (native, python)}
            with NodesetServer(('127.0.0.1', 0), nodesets=nodesets, cache=cache) as server:
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                address = f'http://127.0.0.1:{server.server_port}'

                def get(path):
                    with urlopen(address + path, timeout=20) as response:
                        return json.load(response)

                try:
                    items = get('/api/nodesets')
                    self.assertEqual(len(items), 2)
                    self.assertFalse(cache.exists(), 'Listing should not export every model')
                    for item in items:
                        preflop, postflop = get(item['preflop']), get(item['postflop'])
                        self.assertTrue(preflop)
                        self.assertEqual(postflop, {})
                        exported = cache / item['version'] / 'preflop-model.json'
                        stamp = exported.stat().st_mtime_ns
                        self.assertEqual(get(item['preflop']), preflop)
                        self.assertEqual(exported.stat().st_mtime_ns, stamp, 'Reuse the selected export')
                    request = Request(address + '/api/arena', method='POST',
                                      data=json.dumps({'a': items[0]['id'], 'b': items[1]['id'], 'hands': 20, 'seed': 9}).encode(),
                                      headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=20) as response:
                        match = json.load(response)
                    self.assertEqual(match['hands'], 20)
                    self.assertEqual(match['a']['net_bb'], -match['b']['net_bb'])
                    for bad in ({'a': '../anything', 'b': 'bundled', 'hands': 20},
                                {'a': items[0]['id'], 'b': items[1]['id'], 'hands': 3}):
                        request.data = json.dumps(bad).encode()
                        with self.assertRaises(HTTPError) as failed:
                            urlopen(request, timeout=20)
                        self.assertEqual(failed.exception.code, 400)
                    with self.assertRaises(HTTPError) as failed:
                        get('/api/nodesets/../../FULLGAME_10m_iters.pkl')
                    self.assertEqual(failed.exception.code, 404)
                    first = next(item for item in checkpoints(nodesets) if item['source'] == native.resolve())
                    native.touch()
                    next_version = next(item for item in checkpoints(nodesets) if item['source'] == native.resolve())
                    self.assertEqual(first['id'], next_version['id'], 'Keep the saved selection across updates')
                    self.assertNotEqual(first['version'], next_version['version'])
                finally:
                    server.shutdown()
                    thread.join()
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)


if __name__ == '__main__':
    unittest.main()
