2. Check the /api/tags endpoint:
curl -s "http://127.0.0.1:11434/api/tags" | ConvertFrom-Json

# mediawikirag

Retrieval-Augmented Generation (RAG) system for querying a German MediaWiki instance via a terminal interface. Multiple users can query wiki content using a fast, extractive search pipeline. Index is stored on a shared network drive and updated periodically.

---

## Features
- Fast semantic search over MediaWiki content
- Multi-user support (concurrent queries)
- Configurable search parameters
- Interactive CLI with helpful commands
- Robust error handling and logging

---

## Setup

Install [uv](https://github.com/astral-sh/uv) for dependency management:

```sh
pip install uv
```

Install dependencies:

```sh
uv pip install -r requirements.txt
```

---

## Project Structure

- `src/` — Core modules (fetcher, chunker, embedder, storage, retriever, cli)
- `scripts/` — CLI and update scripts
- `tests/` — Unit and integration tests
- `config.yaml` — Configuration file
- `wiki_rag_architecture.md` — Architecture and design

---

## CLI Usage

Run the interactive query interface:

```sh
python scripts/chat.py
```

### Commands

- **Regular text**: Treated as a search query
- **/help**: Show available commands
- **/info**: Display index metadata (version, stats)
- **/config**: Show/modify search parameters (`top_k`, `similarity_threshold`)
- **/refresh**: Reload index from disk (if updated)
- **/quit** or **/exit**: Exit program

### Example Workflow

```
Welcome to Wiki RAG! Type your question or /help.
> /help
/help - Show commands
/info - Index info
/refresh - Reload index
/config - Search parameters
/quit - Exit
> What is the capital of Germany?
[0.812] Berlin | Geography
		Berlin is the capital and largest city of Germany...
		URL: https://wiki.../Berlin#Geography
----------------------------------------
> /config
Current search parameters:
	top_k: 5
	similarity_threshold: 0.3
Type 'top_k <value>' or 'threshold <value>' to change, or just press Enter to keep.
config> top_k 10
top_k set to 10
> /quit
Goodbye!
```

---

## Configuration

Edit `config.yaml` to set wiki source, storage paths, model, chunking, and retrieval parameters. Example:

```yaml
wiki:
	url: "https://your-wiki.example.com"
	api_endpoint: "/api.php"
storage:
	network_drive: "/network_drive/wiki_rag"
	local_cache: "~/.cache/wiki_rag"
models:
	embedding: "paraphrase-multilingual-MiniLM-L12-v2"
	embedding_dim: 384
	llm:
		enabled: true
		provider: 'ollama'  # use 'gpt4all' or 'ollama'
		model: 'your-ollama-model'
		base_url: 'http://127.0.0.1:11434'
		# request_format controls how the prompt is sent to Ollama.
		# - 'prompt': send a single prompt string in 'prompt'
		# - 'messages': send a chat-style messages list
		# - 'auto': try 'prompt' first and fall back to 'messages' if empty
		request_format: 'auto'
	llm:
		enabled: true
		provider: 'ollama'  # use 'gpt4all' or 'ollama'
		model: 'your-ollama-model'
		base_url: 'http://127.0.0.1:11434'
retrieval:
	top_k: 5
	similarity_threshold: 0.3
```

---

## Troubleshooting

- **"Index not found"**: Run initial `update_index.py --full-rebuild` to create index files.
- **"Lock file exists"**: If updating, check `.update.lock` timestamp. If stale (>2h), delete manually.
- **"Slow startup"**: Enable local caching in config; check network drive speed.
- **"Outdated results"**: Run `/refresh` in chat.py or re-run update script.
- **Errors in CLI**: Check `chat.log` for details.

---

## Architecture & Design

See `wiki_rag_architecture.md` for full system design, update process, storage layout, and technical decisions.

---

## License

MIT