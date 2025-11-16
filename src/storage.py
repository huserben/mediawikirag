                        # map returned chunks back to their embedding indices
                        # map returned chunks back to their embedding indices
                        # map returned chunks back to their embedding indices
# File I/O and locking for Wiki RAG
import json
import numpy as np
from pathlib import Path
import time
import getpass
import socket
import os
import datetime
try:
    # optional import for fast vector search
    from .vectorstore import FaissVectorStore
except Exception:
    FaissVectorStore = None


class Storage:

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.chunks = None
        self.embeddings = None

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
            data = [json.loads(line) for line in f]
        # store in instance for callers that expect attributes
        self.chunks = data
        return data

    def load_embeddings(self, path=None):
        path = path or self.base_path / 'embeddings.npy'
        data = np.load(path)
        # store in instance for callers that expect attributes
        self.embeddings = data
        # try to initialize FAISS index if available. Use importlib to attempt
        # absolute imports first so this works when `src` is added to sys.path
        # (how scripts are executed in this repo).
        try:
            import importlib
            _LocalFaiss = None
            try:
                mod = importlib.import_module('vectorstore')
                _LocalFaiss = getattr(mod, 'FaissVectorStore', None)
            except Exception:
                try:
                    mod = importlib.import_module('.vectorstore', package=__package__)
                    _LocalFaiss = getattr(mod, 'FaissVectorStore', None)
                except Exception:
                    _LocalFaiss = None

            if _LocalFaiss is not None:
                self._vectorstore = _LocalFaiss(self.base_path)
                try:
                    # prefer loading prebuilt index
                    self._vectorstore.load()
                except Exception:
                    # build from embeddings in-memory
                    try:
                        self._vectorstore.build_index(self.embeddings)
                        # attempt to save, but ignore failures
                        try:
                            self._vectorstore.save()
                        except Exception:
                            pass
                    except Exception:
                        # if faiss build fails, remove vectorstore
                        self._vectorstore = None
        except Exception:
            self._vectorstore = None
        # try to initialize Chroma index if available
        try:
            import importlib
            _ChromaCls = None
            try:
                mod = importlib.import_module('vectorstore_chroma')
                _ChromaCls = getattr(mod, 'ChromaVectorStore', None)
            except Exception:
                try:
                    mod = importlib.import_module('.vectorstore_chroma', package=__package__)
                    _ChromaCls = getattr(mod, 'ChromaVectorStore', None)
                except Exception:
                    _ChromaCls = None

            if _ChromaCls is not None:
                try:
                    self._chroma = _ChromaCls(self.base_path)
                    try:
                        self._chroma.load()
                    except Exception:
                        # nothing to do; collection will be available after update
                        pass
                except Exception:
                    self._chroma = None
            else:
                self._chroma = None
        except Exception:
            self._chroma = None
        return data

    def search(self, query_emb, top_k=5, threshold=0.3,
               use_mmr=False, fetch_k=20, lambda_param=0.5):
        """Search for relevant chunks using FAISS if available, else fallback.

        Returns a list of (chunk_dict, score) tuples sorted by descending score.
        """
        # Prefer Chroma if present
        if hasattr(self, '_chroma') and getattr(self, '_chroma', None) is not None:
            try:
                ids, scores = self._chroma.search(query_emb, top_k=fetch_k if use_mmr else top_k)
                # Chroma may return distances; treat them as scores for now
                results = []
                for idx, score in zip(ids, scores):
                    try:
                        chunk = self.chunks[int(idx)]
                    except Exception:
                        continue
                    if score >= threshold:
                        results.append((chunk, float(score)))
                # If MMR requested, rerank using full embeddings
                if use_mmr and len(results) > 0:
                    try:
                        from .retriever import mmr_rerank
                        # collect candidate embeddings and chunks
                        cand_ids = [self.chunks.index(c) for c, _ in results]
                        cand_embs = np.array([self.embeddings[idx] for idx in cand_ids])
                        cand_chunks = [c for c, _ in results]
                        return mmr_rerank(query_emb, cand_embs, cand_chunks, top_k=top_k,
                                          fetch_k=fetch_k, lambda_param=lambda_param)
                    except Exception:
                        return results[:top_k]
                return results[:top_k]
            except Exception:
                pass

        # Prefer FAISS-backed search
        if hasattr(self, '_vectorstore') and getattr(self, '_vectorstore', None) is not None:
            try:
                ids, scores = self._vectorstore.search(query_emb, top_k=fetch_k if use_mmr else top_k)
                results = []
                for idx, score in zip(ids, scores):
                    try:
                        chunk = self.chunks[int(idx)]
                    except Exception:
                        continue
                    if score >= threshold:
                        results.append((chunk, float(score)))
                # MMR rerank if requested
                if use_mmr and len(results) > 0:
                    try:
                        from .retriever import mmr_rerank
                        cand_ids = [self.chunks.index(c) for c, _ in results]
                        cand_embs = np.array([self.embeddings[idx] for idx in cand_ids])
                        cand_chunks = [c for c, _ in results]
                        return mmr_rerank(query_emb, cand_embs, cand_chunks, top_k=top_k,
                                          fetch_k=fetch_k, lambda_param=lambda_param)
                    except Exception:
                        return results[:top_k]
                return results[:top_k]
            except Exception:
                pass

        # Fallback: brute-force cosine similarity using stored embeddings
        try:
            from .retriever import search_chunks, mmr_rerank
            if use_mmr:
                # fetch a larger candidate set then rerank
                candidates = search_chunks(query_emb, self.embeddings, self.chunks, top_k=fetch_k, threshold=threshold)
                if not candidates:
                    return []
                cand_embs = np.array([self.embeddings[self.chunks.index(c)] for c, _ in candidates])
                cand_chunks = [c for c, _ in candidates]
                return mmr_rerank(query_emb, cand_embs, cand_chunks, top_k=top_k,
                                  fetch_k=fetch_k, lambda_param=lambda_param)
            else:
                return search_chunks(query_emb, self.embeddings, self.chunks, top_k=top_k, threshold=threshold)
        except Exception:
            return []

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
