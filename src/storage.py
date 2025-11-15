# File I/O and locking for Wiki RAG
import json
import numpy as np
from pathlib import Path
import time
import getpass
import socket
import os
import datetime


class Storage:

    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def save_chunks(self, chunks, path=None):
        path = path or self.base_path / 'chunks.jsonl'
        tmp_path = str(path) + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        Path(tmp_path).replace(path)

    def save_embeddings(self, embeddings, path=None):
        path = path or self.base_path / 'embeddings.npy'
        tmp_path = str(path) + '.tmp'
        np.save(tmp_path, embeddings)
        # np.save creates .npy file, so rename explicitly
        tmp_npy_path = tmp_path if tmp_path.endswith('.npy') else tmp_path + '.npy'
        if not Path(tmp_npy_path).exists():
            # np.save may have created tmp_path+'.npy' instead of tmp_path
            tmp_npy_path = tmp_path + '.npy'
        Path(tmp_npy_path).replace(path)

    def load_chunks(self, path=None):
        path = path or self.base_path / 'chunks.jsonl'
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def load_embeddings(self, path=None):
        path = path or self.base_path / 'embeddings.npy'
        return np.load(path)

    def acquire_lock(self, lock_path=None, timeout=7200):
        lock_path = lock_path or self.base_path / '.update.lock'
        if Path(lock_path).exists():
            with open(lock_path, 'r', encoding='utf-8') as f:
                lock_info = json.load(f)
            lock_time = time.mktime(
                time.strptime(lock_info['timestamp'], '%Y-%m-%dT%H:%M:%S')
            )
            if time.time() - lock_time > timeout:
                # Stale lock, override
                pass
            else:
                raise RuntimeError(
                    f"Lock held by {lock_info.get('username', 'unknown')}"
                )
        # Write new lock
        info = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'username': getpass.getuser(),
            'pid': os.getpid(),
            'hostname': socket.gethostname()
        }
        with open(lock_path, 'w', encoding='utf-8') as f:
            json.dump(info, f)

    def release_lock(self, lock_path=None):
        lock_path = lock_path or self.base_path / '.update.lock'
        if Path(lock_path).exists():
            Path(lock_path).unlink()
        if Path(lock_path).exists():
            Path(lock_path).unlink()
