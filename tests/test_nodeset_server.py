import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from cpp.run import build
import pf_mccfr
from tools.serve import NodesetServer, checkpoints


class NodesetServerTests(unittest.TestCase):
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
