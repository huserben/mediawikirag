# Search/similarity logic for Wiki RAG

import numpy as np


def cosine_similarity(query_emb, chunk_embs):
    """Compute cosine similarity between a single query embedding and
    an array-like of chunk embeddings.

    Returns a 1-D numpy array of scores with the same length as
    `chunk_embs`.
    """
    query_emb = np.array(query_emb)
    chunk_embs = np.array(chunk_embs)
    if chunk_embs.size == 0:
        return np.array([])
    dot = np.dot(chunk_embs, query_emb)
    norm_query = np.linalg.norm(query_emb)
    norm_chunks = np.linalg.norm(chunk_embs, axis=1)
    # protect against zero division
    denom = norm_query * norm_chunks
    denom[denom == 0] = 1e-12
    return dot / denom


def search_chunks(query_emb, chunk_embs, chunks, top_k=5, threshold=0.3):
    """Return up to `top_k` chunks with cosine similarity >= `threshold`.

    Args:
        query_emb: list/ndarray for the query embedding.
        chunk_embs: list/ndarray of shape (N, D) with stored embeddings.
        chunks: list of chunk dicts aligned with `chunk_embs` ordering.
        top_k: maximum number of results to return.
        threshold: minimum cosine score to include a result.

    Returns:
        List of (chunk_dict, float(score)) sorted by descending score.
    """
    # Defensive checks
    if chunk_embs is None or len(chunk_embs) == 0 or not chunks:
        return []

    scores = cosine_similarity(query_emb, chunk_embs)
    if scores.size == 0:
        return []

    # Get indices sorted by descending score
    order = np.argsort(scores)[::-1]
    results = []
    for idx in order:
        if len(results) >= top_k:
            break
        score = float(scores[idx])
        if score >= threshold:
            try:
                results.append((chunks[int(idx)], score))
            except Exception:
                # index mismatch or bad chunk list - skip
                continue
    return results


def mmr_rerank(query_emb, candidate_embs, candidate_chunks, top_k=5, fetch_k=20, lambda_param=0.5):
    """Perform Maximal Marginal Relevance (MMR) reranking.

    Args:
        query_emb: 1-D array-like query embedding.
        candidate_embs: 2-D array-like candidate embeddings (N x D).
        candidate_chunks: list of chunk dicts aligned with candidate_embs.
        top_k: number of final results to return.
        fetch_k: number of top candidates to consider before MMR (speed/quality tradeoff).
        lambda_param: trade-off parameter between relevance and diversity (0..1).

    Returns:
        List of (chunk, score) tuples of length <= top_k.
    """
    if candidate_embs is None or len(candidate_embs) == 0:
        return []

    embs = np.array(candidate_embs)
    q = np.array(query_emb)
    # compute cosine similarities
    sims = cosine_similarity(q, embs)
    # consider top fetch_k candidates by similarity
    order = np.argsort(sims)[::-1][:fetch_k]
    candidate_ids = list(order)

    selected = []  # indices
    selected_scores = []

    # precompute pairwise similarities among candidates
    if len(candidate_ids) > 0:
        cand_embs_top = embs[candidate_ids]
        pairwise = np.matmul(cand_embs_top, cand_embs_top.T)
    else:
        pairwise = np.array([[]])

    # mapping from local top-k index to global index
    for _ in range(min(top_k, len(candidate_ids))):
        if not selected:
            # pick the most similar to query first
            best_local = 0
            best_global = candidate_ids[best_local]
            selected.append(best_global)
            selected_scores.append(float(sims[best_global]))
            continue

        best_score = None
        best_global = None
        for rank_pos, global_idx in enumerate(candidate_ids):
            if global_idx in selected:
                continue
            rel = float(sims[global_idx])
            # compute redundancy as max similarity to already selected
            # need local index into cand_embs_top
            try:
                local_pos = candidate_ids.index(global_idx)
            except ValueError:
                continue
            if selected:
                # compute max similarity to any selected (using pairwise)
                sel_local_positions = [candidate_ids.index(s) for s in selected]
                redundancy = max(pairwise[local_pos, pos] for pos in sel_local_positions)
            else:
                redundancy = 0.0
            mmr_score = lambda_param * rel - (1 - lambda_param) * redundancy
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_global = global_idx

        if best_global is None:
            break
        selected.append(best_global)
        selected_scores.append(float(sims[best_global]))

    # return corresponding chunks with original cosine scores
    results = []
    for idx, score in zip(selected, selected_scores):
        try:
            results.append((candidate_chunks[int(idx)], score))
        except Exception:
            continue
    return results
