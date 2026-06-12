# SEC Domain

Generates multi-hop search tasks from SEC filings (10-K, 20-F, etc.) stored in ChromaDB. An LLM agent explores chunked filings via semantic search, formulates a question with 3 supporting chunks, then extends with cross-company bridging hops for N rounds.

## Setup

```bash
uv sync --extra sec-index --extra rerank
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CHROMA_API_KEY` | ChromaDB Cloud API key |
| `CHROMA_DATABASE` | ChromaDB Cloud database name |
| `OPENAI_API_KEY` | OpenAI API key (embeddings) |
| `BASETEN_API_KEY` | Baseten API key (reranking) |
| `ANTHROPIC_API_KEY` | Anthropic API key (LLM calls) |

## Quick Start

```bash
python -m agentic_search_data_gen.domains.sec \
  -o data/sec/output -c my_collection --identity "Name email@example.com"
```

Uses the default seeds at `agentic_search_data_gen/domains/sec/seeds.txt`. Runs the full pipeline: index → assign truth types → explore → verify → collect → verify-collect, then repeats extend → verify-extension for `--extension-rounds` rounds (default: 1).

Run multiple extension rounds for deeper cross-company chaining:

```bash
python -m agentic_search_data_gen.domains.sec \
  -o data/sec/output -c my_collection --identity "Name email@example.com" \
  --extension-rounds 3
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds/-s` | `seeds.txt` (in module dir) | Path to seeds file (one ticker per line) |
| `--output/-o` | **required** | Output directory |
| `--collection/-c` | **required** | ChromaDB collection name |
| `--max-workers/-w` | 8 | Parallel workers |
| `--extension-rounds` | 1 | Number of extend → verify rounds |
| `--no-index` | false | Skip ChromaDB indexing stage |
| `--identity` | **required** | SEC EDGAR identity (e.g. `"Name email@example.com"`) |
| `--explore-model` | `claude-sonnet-4-5` | Model for explore stage |
| `--explore-max-iterations` | 20 | Max iterations for explore |
| `--verify-model` | `claude-opus-4-5` | Model for verify stages |
| `--verify-max-retries` | 3 | Max retries for verify |
| `--collect-model` | `claude-sonnet-4-5` | Model for collect stage |
| `--collect-max-iterations` | 15 | Max iterations for collect |
| `--extend-agent-model` | `claude-sonnet-4-5` | Model for extend agent |
| `--extend-verification-model` | `claude-opus-4-5` | Model for extend verification |
| `--extend-max-iterations-phase1` | 10 | Max iterations for extend phase 1 |
| `--extend-max-iterations-phase2` | 15 | Max iterations for extend phase 2 |

## Pipeline

The unified command chains these stages automatically. Stages 1–6 run once, then stages 7–8 repeat for each `--extension-rounds` round.

```
1. Index          Download, chunk, and index filings into ChromaDB
2. Assign types   Assign random truth_type to each ticker JSON
3. Explore        Generate a multi-hop question per company
4. Verify         Extract and verify quotes from supporting items
5. Collect        Find additional corroborating chunks
6. Verify collect Verify the additional chunks
                  ┌─── repeat for each extension round ───┐
7. Extend         │ Cross-company bridging hop             │
8. Verify ext     │ Verify the extended task               │
                  └───────────────────────────────────────-┘
```

### Individual stages

Each stage can also be run standalone:

**Index**
```bash
python -m agentic_search_data_gen.domains.sec.index \
  -i agentic_search_data_gen/domains/sec/seeds.txt -o data/sec/output \
  -w 8 --collection sec_filings --identity "Name email@example.com"
```
Downloads filings, chunks them, indexes into ChromaDB with OpenAI dense + BM25 sparse embeddings. Skips tickers with existing output files. Pass `--no-index` to save chunks locally without uploading.

**Explore**
```bash
python -m agentic_search_data_gen.domains.sec.explore \
  -i data/sec/output -w 8 -n 20 --collection sec_filings --model claude-sonnet-4-5
```
Generates a task per company from its indexed filings. Requires `truth_type` in each JSON.

**Verify**
```bash
python -m agentic_search_data_gen.domains.sec.verify \
  -i data/sec/output -w 8 -m claude-opus-4-5
```
Extracts and verifies quotes from supporting items. Use `--mode collect` for additional chunks.

**Collect**
```bash
python -m agentic_search_data_gen.domains.sec.collect \
  -i data/sec/output -w 4 -n 15 --collection sec_filings --model claude-sonnet-4-5
```
Finds additional chunks corroborating each supporting item.

**Extend**
```bash
python -m agentic_search_data_gen.domains.sec.extend \
  -i data/sec/output -w 8 --collection sec_filings \
  --agent-model claude-sonnet-4-5 --verification-model claude-opus-4-5
```
Creates cross-company bridging hops. Appends a `level=1` task with `bridging_item` and `new_company`.

## Output format

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "available_forms": {"10-K": ["0000320193-24-000081"]},
  "tasks": [
    {
      "level": 0,
      "clues": "...",
      "question": "...",
      "truth": "...",
      "truth_type": "...",
      "supporting_items": [
        {"id": "chunk_abc123", "clue_quotes": [], "item_quotes": [], "contains_truth": true, "additional_chunks": []}
      ],
      "items_and_contents": {"chunk_abc123": "chunk text..."}
    }
  ]
}
```
