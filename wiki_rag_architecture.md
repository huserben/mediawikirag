# Wiki RAG System - Architecture Document

## Project Overview

A Python-based Retrieval-Augmented Generation (RAG) system for querying a German MediaWiki instance. The system allows multiple users to query the wiki content via a terminal interface, with periodic updates to the index stored on a shared network drive.

### Key Requirements
- **Multi-user support**: Multiple users can query simultaneously
- **Network storage**: Index stored on shared network drive
- **Coordinated updates**: Prevent concurrent update conflicts
- **German language**: Optimized for German wiki content
- **Non-root environment**: Ubuntu without sudo/root access
- **Simple UX**: Terminal-based, accessible to non-coders

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Network Drive Storage                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ /network_drive/wiki_rag/                               │ │
│  │   ├── .update.lock          (coordination)             │ │
│  │   ├── current/              (stable read-only index)   │ │
│  │   ├── staging/              (update workspace)         │ │
│  │   └── archive/              (previous versions)        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↑
                            │ (read/write)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼─────┐       ┌─────▼────┐       ┌─────▼────┐
   │  User 1  │       │  User 2  │       │ Updater  │
   │ (query)  │       │ (query)  │       │ (write)  │
   └──────────┘       └──────────┘       └──────────┘
```

### Two-Phase System Design

#### Phase 1: Indexing (update_index.py)
Run manually, periodically (weekly/monthly)
- Fetches pages from MediaWiki API
- Chunks text content semantically
- Generates embeddings using sentence-transformers
- Stores to network drive with locking

#### Phase 2: Querying (chat.py)
Run by end-users, multiple concurrent instances
- Loads pre-computed index from network drive
- Accepts natural language questions
- Retrieves relevant chunks via similarity search
- Displays context to user (extractive approach)

---

## Data Storage Design

### Directory Structure

```
/network_drive/wiki_rag/
│
├── .update.lock                    # Lock file for update coordination
│   └── Contains: timestamp, username, PID
│
├── current/                        # Active index (read-only for queries)
│   ├── chunks.jsonl               # Text chunks with metadata
│   ├── embeddings.npy             # Numpy array of embeddings
│   ├── metadata.json              # Index version, model info, stats
│   └── wiki_state.json            # Page revisions for change detection
│
├── staging/                        # Temporary update workspace
│   └── [same structure as current/]
│
├── archive/                        # Historical versions
│   ├── 2025-11-01_14-30/
│   ├── 2025-11-08_09-15/
│   └── ...
│
└── logs/
    └── updates.log                # Update history and errors
```

### File Formats

#### chunks.jsonl
```json
{"id": "page123_chunk001", "text": "...", "page_title": "...", "section": "...", "url": "..."}
{"id": "page123_chunk002", "text": "...", "page_title": "...", "section": "...", "url": "..."}
```

#### embeddings.npy
- Numpy array: shape (n_chunks, embedding_dim)
- Float32 for memory efficiency
- Row i corresponds to chunk i in chunks.jsonl

#### metadata.json
```json
{
  "version": "2025-11-15T10:30:00",
  "wiki_url": "https://your-wiki.example.com",
  "total_pages": 342,
  "total_chunks": 8432,
  "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dim": 384,
  "chunk_size": 800,
  "chunk_overlap": 150
}
```

#### wiki_state.json
```json
{
  "pages": {
    "Page Title 1": {"revid": 12345, "timestamp": "2025-11-01T10:00:00"},
    "Page Title 2": {"revid": 12346, "timestamp": "2025-11-02T14:30:00"}
  }
}
```

---

## Update Process (update_index.py)

### Update Flow

```
1. Check/Acquire Lock
   ↓
2. Fetch Changed Pages (MediaWiki API)
   ↓
3. Process: Chunk + Embed
   ↓
4. Write to staging/
   ↓
5. Validate Integrity
   ↓
6. Atomic Swap: staging/ → current/
   ↓
7. Archive Old Version
   ↓
8. Release Lock
```

### Lock Mechanism

**Lock File Structure (.update.lock):**
```json
{
  "timestamp": "2025-11-15T10:30:00",
  "username": "alice",
  "pid": 12345,
  "hostname": "workstation-05"
}
```

**Lock Logic:**
1. Check if `.update.lock` exists
2. If exists:
   - Read timestamp
   - If older than 2 hours → assume stale, override
   - Otherwise → exit with error message
3. Create lock file with current info
4. Perform update in try/finally block
5. Always delete lock in finally

### Change Detection Strategy

**Incremental Updates:**
1. Load `wiki_state.json` from current index
2. Query MediaWiki API: `action=query&generator=allpages&prop=revisions`
3. Compare revision IDs
4. Identify: new pages, modified pages, deleted pages
5. Only re-process changed content
6. Merge with existing chunks

**Full Rebuild Option:**
- Command flag: `--full-rebuild`
- Ignores existing state, re-fetches everything
- Useful for model changes or major wiki restructuring

### Error Handling in Updates

- **Network failures**: Retry logic with exponential backoff
- **Partial updates**: Rollback to previous version on error
- **Lock conflicts**: Clear error message, show who holds lock
- **Validation failures**: Checksum mismatches → abort, log error

---

## Query Process (chat.py)

### Startup Sequence

```
1. Check if current/ exists
   ↓
2. Load metadata.json (display version info)
   ↓
3. Load chunks.jsonl into memory
   ↓
4. Load embeddings.npy into memory
   ↓
5. Initialize embedding model
   ↓
6. Start interactive loop
```

### Query Flow

```
User Question
   ↓
Embed Question (sentence-transformers)
   ↓
Compute Cosine Similarity (with all chunk embeddings)
   ↓
Retrieve Top-K Chunks (k=5, configurable)
   ↓
Display Results:
   - Chunk text
   - Source page + section
   - Similarity score
   - URL for reference
```

### Interactive Commands

- **Regular text**: Treated as query
- **/help**: Show available commands
- **/info**: Display index metadata (version, stats)
- **/refresh**: Reload index from disk (if update detected)
- **/config**: Show/modify search parameters (top-k, threshold)
- **/quit** or **/exit**: Exit program

### Concurrent Access Pattern

- Each user runs separate Python process
- All processes read from `current/` (read-only)
- No inter-process communication needed
- Network drive handles file-level read concurrency
- Memory isolation: each process has own copy in RAM

---

## Technical Stack

### Core Dependencies

```
sentence-transformers>=2.2.0    # Embedding generation
requests>=2.31.0                # MediaWiki API
numpy>=1.24.0                   # Embedding storage
tqdm>=4.66.0                    # Progress bars
pyyaml>=6.0                     # Config file parsing
```

### Optional Dependencies

```
rich>=13.0.0                    # Enhanced terminal UI
```

### Python Version
- Minimum: Python 3.9
- Recommended: Python 3.10+

---

## Model Selection

### Embedding Model (German-optimized)

**Primary Choice:**
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embedding dimension: 384
- Supports: 50+ languages including German
- Performance: Good balance of quality/speed
- Size: ~120MB

**Alternative (German-specific):**
- Model: `deutsche-telekom/gbert-large-paraphrase-cosine`
- Better German quality, larger size
- Consider if multilingual model underperforms

### No Generation Model (Extractive Only)

**Decision**: Start with extractive retrieval only
- Return relevant wiki chunks verbatim
- User reads context themselves
- Simpler, faster, more reliable
- Better for source verification
- Can add generation later if needed (e.g., local Mistral or API-based)

---

## Text Processing

### Chunking Strategy

**Semantic Chunking (Section-based):**
1. Parse MediaWiki markup to identify sections (== Heading ==)
2. Split by sections first
3. If section too large (>1000 tokens):
   - Split by paragraphs
   - If still too large, split by sentences
4. Maintain chunk overlap: 150 characters
5. Preserve metadata: page title, section name, hierarchy

**Chunk Metadata:**
```python
{
    "id": "unique_chunk_id",
    "text": "chunk content...",
    "page_title": "Original Page Title",
    "section": "Section > Subsection",
    "url": "https://wiki.../Page_Title#Section",
    "chunk_index": 2,  # nth chunk from this page
    "char_count": 784
}
```

### Preprocessing Steps

1. Fetch raw MediaWiki content
2. Convert markup to plain text (basic parsing, keep structure)
3. Remove: templates, table markup, image links
4. Keep: headings, paragraphs, lists, basic formatting
5. Normalize: whitespace, encoding
6. Chunk as described above
7. Generate embeddings

---

## Performance Optimization

### Network Drive Mitigation

**Strategy: Local Caching**
- Each user maintains `~/.cache/wiki_rag/`
- On startup: check metadata.json hash
- If hash matches → use local cache
- If hash differs → re-download from network
- Dramatically speeds up startup after first run

**Implementation:**
```python
# Pseudo-code
cache_dir = Path.home() / ".cache" / "wiki_rag"
network_metadata = load_json(network_path / "metadata.json")
local_metadata = load_json(cache_dir / "metadata.json") if exists

if network_metadata["version"] != local_metadata.get("version"):
    print("New index version detected, downloading...")
    copy_to_cache(network_path / "current", cache_dir)
else:
    print("Using cached index")

# Load from cache_dir instead of network_path
```

### Memory Optimization

**Embedding Storage:**
- Use float32 instead of float64 (half the size)
- Estimate: 384-dim embeddings = ~1.5KB per chunk
- 10,000 chunks = ~15MB embeddings
- Acceptable for in-memory operation

**Lazy Loading Option:**
- If wiki is huge (>50k chunks), consider:
  - Load metadata on startup
  - Load embeddings on first query
  - Show progress indicator

### Search Performance

**Cosine Similarity:**
- Numpy vectorized operations: very fast
- 10k chunks: <100ms on typical CPU
- No need for approximate search (FAISS) at this scale
- If wiki grows >100k chunks, revisit with FAISS

---

## Configuration Management

### config.yaml

```yaml
# MediaWiki Source
wiki:
  url: "https://your-wiki.example.com"
  api_endpoint: "/api.php"
  
# Storage Paths
storage:
  network_drive: "/network_drive/wiki_rag"
  local_cache: "~/.cache/wiki_rag"
  
# Model Settings
models:
  embedding: "paraphrase-multilingual-MiniLM-L12-v2"
  embedding_dim: 384
  
# Chunking Parameters
chunking:
  target_size: 800        # characters
  overlap: 150            # characters
  min_size: 200           # minimum chunk size
  max_size: 1500          # maximum chunk size
  
# Retrieval Settings
retrieval:
  top_k: 5                # number of chunks to retrieve
  similarity_threshold: 0.3  # minimum similarity score
  
# Update Settings
update:
  lock_timeout: 7200      # seconds (2 hours)
  archive_keep: 5         # number of old versions to keep
  batch_size: 50          # pages to process per batch
  
# Logging
logging:
  level: "INFO"
  file: "logs/updates.log"
```

---

## Project Structure

```
wiki_rag/
│
├── config.yaml                    # Configuration
├── requirements.txt               # Python dependencies
├── README.md                      # User documentation
├── ARCHITECTURE.md                # This document
│
├── scripts/
│   ├── setup.sh                   # Initial setup script
│   ├── update_index.py           # Main update script
│   └── chat.py                    # Main query script
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loading
│   ├── fetcher.py                 # MediaWiki API interaction
│   ├── chunker.py                 # Text chunking logic
│   ├── embedder.py                # Embedding generation
│   ├── storage.py                 # File I/O, locking
│   ├── retriever.py               # Search/similarity logic
│   └── cli.py                     # Terminal UI helpers
│
└── tests/
    ├── test_fetcher.py
    ├── test_chunker.py
    ├── test_storage.py
    └── test_retriever.py
```

---

## Implementation Phases

### Phase 1: Core Indexing (MVP)
- [ ] MediaWiki API fetcher (all pages, basic)
- [ ] Text chunking (simple fixed-size first)
- [ ] Embedding generation
- [ ] Storage to JSONL + numpy
- [ ] Test with 5-10 pages

### Phase 2: Query Interface
- [ ] Load index into memory
- [ ] Embedding query text
- [ ] Cosine similarity search
- [ ] Display top-k results
- [ ] Basic CLI loop

### Phase 3: Multi-user Support
- [ ] Lock file mechanism
- [ ] Staging/current/archive structure
- [ ] Atomic swap logic
- [ ] Error handling and rollback

### Phase 4: Incremental Updates
- [ ] wiki_state.json tracking
- [ ] Change detection via MediaWiki API
- [ ] Merge updated chunks with existing
- [ ] Handle page deletions

### Phase 5: Polish & Optimization
- [ ] Local caching layer
- [ ] Semantic chunking (section-based)
- [ ] Rich terminal UI
- [ ] Config file support
- [ ] Comprehensive error messages
- [ ] Logging and monitoring

### Phase 6: Optional Enhancements
- [ ] Generation model integration
- [ ] Query history
- [ ] Statistics/analytics
- [ ] Web UI (optional)

---

## Testing Strategy

### Unit Tests
- **Chunker**: Various text inputs, edge cases
- **Embedder**: Mock model, shape verification
- **Storage**: Lock acquisition, atomic operations
- **Retriever**: Similarity calculation correctness

### Integration Tests
- **End-to-end indexing**: Small test wiki
- **Concurrent queries**: Simulate multiple users
- **Update during queries**: Verify no corruption
- **Lock conflicts**: Two updaters simultaneously

### Manual Testing Checklist
- [ ] Index creation from scratch
- [ ] Query accuracy (sample questions)
- [ ] Update with no changes (no-op)
- [ ] Update with new pages
- [ ] Update with modified pages
- [ ] Update with deleted pages
- [ ] Lock timeout handling
- [ ] Network drive disconnection
- [ ] Cache invalidation
- [ ] Stale lock override

---

## Deployment Guide

### Initial Setup (One-time)

```bash
# 1. Clone/copy project to shared location
cd /network_drive/
mkdir -p wiki_rag
cd wiki_rag

# 2. Create directory structure
mkdir -p current staging archive logs

# 3. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download embedding model (pre-cache)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 6. Configure
cp config.yaml.example config.yaml
# Edit config.yaml with your wiki URL

# 7. Initial index build
python scripts/update_index.py --full-rebuild
```

### User Setup (Per User)

```bash
# 1. Add to .bashrc or create wrapper script
cat > ~/bin/wiki-chat << 'EOF'
#!/bin/bash
cd /network_drive/wiki_rag
source venv/bin/activate
python scripts/chat.py
EOF

chmod +x ~/bin/wiki-chat

# 2. Run
wiki-chat
```

### Update Process (Periodic)

```bash
# Option 1: Manual
cd /network_drive/wiki_rag
source venv/bin/activate
python scripts/update_index.py

# Option 2: Cron job (if available)
# Add to crontab:
# 0 2 * * 0 cd /network_drive/wiki_rag && source venv/bin/activate && python scripts/update_index.py
```

---

## Monitoring & Maintenance

### Health Checks
- Check `logs/updates.log` for errors
- Verify last update timestamp in metadata.json
- Monitor archive/ size (cleanup old versions)
- Check for stale lock files

### Common Issues & Solutions

**"Index not found"**
- Solution: Run initial `update_index.py --full-rebuild`

**"Lock file exists"**
- Check timestamp, if >2 hours → `rm .update.lock`
- Or use: `update_index.py --force`

**"Slow query startup"**
- Enable local caching in config
- Check network drive latency

**"Outdated results"**
- Run `/refresh` in chat.py
- Or manually run update_index.py

---

## Future Enhancements (Post-MVP)

### Short-term
- **Automatic update detection**: Chat.py checks for new version on startup
- **Query refinement**: Follow-up questions with context
- **Highlighting**: Show matched keywords in results
- **Export results**: Save query results to file

### Medium-term
- **Generation mode**: Summarize retrieved chunks using LLM
- **Multi-wiki support**: Query across multiple wikis
- **Query history**: Track and revisit past queries
- **Statistics dashboard**: Most queried topics, coverage analysis

### Long-term
- **Web UI**: Browser-based interface (Flask/FastAPI)
- **Advanced retrieval**: Hybrid search (keyword + semantic)
- **Feedback loop**: User ratings to improve ranking
- **Real-time updates**: WebSocket notifications for index updates

---

## Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| JSONL + numpy for storage | Simple, readable, good performance at scale | 2025-11-15 |
| Extractive only (no generation) | Simpler, reliable, verifiable sources | 2025-11-15 |
| Lock file coordination | Simple, no additional infrastructure | 2025-11-15 |
| Local caching layer | Mitigate network drive latency | 2025-11-15 |
| Multilingual model | Good German support, future flexibility | 2025-11-15 |
| Semantic chunking | Better context preservation than fixed-size | 2025-11-15 |

---

## Questions for Implementation

Before starting implementation, clarify:

1. **Wiki size**: Approximate number of pages? Total text volume?
2. **Update frequency**: How often does content change? (daily/weekly/monthly)
3. **User count**: How many concurrent users expected?
4. **Network drive speed**: Rough latency for file access?
5. **MediaWiki API access**: Any authentication required? Rate limits?
6. **Query patterns**: Typical questions users will ask? (helps test retrieval quality)

---

## Success Criteria

### Must Have (MVP)
- ✅ Index creation from MediaWiki works
- ✅ Multiple users can query simultaneously
- ✅ Updates don't corrupt index
- ✅ Relevant chunks returned for test queries
- ✅ Simple terminal interface works

### Should Have
- ✅ Incremental updates (not full rebuild each time)
- ✅ Local caching improves performance
- ✅ Lock mechanism prevents conflicts
- ✅ Error messages are clear and actionable

### Nice to Have
- ✅ Rich terminal UI with colors/formatting
- ✅ Query refinement and history
- ✅ Statistics and monitoring
- ✅ Automated update scheduling

---

*Document Version: 1.0*  
*Last Updated: 2025-11-15*  
*Status: Ready for Implementation*