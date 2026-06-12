# Email (Epstein) Domain

Generates multi-hop search tasks from email thread collections stored in ChromaDB. An LLM agent explores threads via hybrid search (semantic + keyword), grep, and random sampling, then formulates a question with 3 supporting threads.

## Setup

```bash
uv sync --extra epstein --extra indexing
```

## Data Sources

Data is hosted on [Google Drive](https://drive.google.com/drive/folders/1Ma_xndJlcB7vBfZSi4yyBiWKOIfZYTRb?usp=sharing) and downloaded automatically by the indexing script.

Two files:

| File | Description | Used for |
|------|-------------|----------|
| `epstein_only.json` | Epstein-only deduplicated threads | **Eval generation** (this pipeline) |
| `enron_and_epstein_chunks.json` | Epstein + Enron emails combined | **Eval corpus** (agent searches this at eval time) |

The Enron emails ([CMU Enron corpus](https://www.cs.cmu.edu/~enron/)) are added to the eval corpus so the agent has a larger, noisier search space. Eval generation uses only Epstein threads so that ground-truth answers come exclusively from the Epstein dataset.

### Upstream data prep (already done)

The raw Epstein emails ([HuggingFace](https://huggingface.co/datasets/notesbymuneeb/epstein-emails)) went through:

1. CSV parsing of JSON-serialized email lists
2. Empty-body removal
3. Formatting with From/To/Date/Subject headers
4. Levenshtein + semantic deduplication

The output is `epstein_only.json` — a dict mapping thread ID to pre-formatted thread text.

## Pipeline

```
download epstein_only.json → chunk → index → explore → verify
```

### Unified Command

```bash
python -m agentic_search_data_gen.domains.epstein \
  -o data/epstein/output \
  -c my_collection
```

Options:
- `--input / -i` — Path to `epstein_only.json` (default: download from Google Drive)
- `--output / -o` — Output directory (required)
- `--collection / -c` — ChromaDB collection name (required)
- `--num / -n` — Number of explorations (default: 10)
- `--max-workers / -w` — Parallel workers (default: 8)
- `--max-iterations` — Max iterations per exploration (default: 20)
- `--explore-model` — Model for exploration (default: `claude-sonnet-4-5`)
- `--verify-model` — Model for verification (default: `claude-opus-4-5`)
- `--embedding-model` — OpenAI embedding model (default: `text-embedding-3-small`)

### Individual Stages

#### 1. Index (download → chunk → index)

```bash
python -m agentic_search_data_gen.domains.epstein.index \
  -c epstein_v1 -o data/epstein
```

Downloads `epstein_only.json` from Google Drive (skipped if already present), chunks threads into ≤512 token pieces, saves chunks JSON, and indexes into ChromaDB Cloud with BM25 sparse + dense vector indexes.

Pass `--input / -i` to use a local file instead of downloading.

#### 2. Explore

Generate tasks from email threads. The agent receives random seed threads and uses search tools to find connections.

```bash
python -m agentic_search_data_gen.domains.epstein.explore \
  -n 10 -o data/epstein/output -w 8 -i 20 -m claude-sonnet-4-5
```

**Output:** One JSON per exploration containing `tasks[0]` with `clues`, `question`, `truth`, `supporting_items`, and `items_and_contents`.

#### 3. Verify

Extract and verify quotes from supporting threads, then run a coherence check.

```bash
python -m agentic_search_data_gen.domains.epstein.verify \
  -i data/epstein/output -w 8 -m claude-opus-4-5
```

Two-step verification:
1. **Quote extraction:** Verifies clue quotes, item quotes, and truth quotes match the thread content.
2. **Coherence check:** An LLM judge verifies the supporting items connect logically to the question and truth.

## Output Format

```json
{
  "tasks": [
    {
      "clues": "...",
      "question": "...",
      "truth": "...",
      "truth_type": "...",
      "supporting_items": [
        {"id": "thread_123", "clue_quotes": [...], "item_quotes": [...], "contains_truth": true}
      ],
      "items_and_contents": {"thread_123": "thread text..."},
      "passed_verification": true,
      "coherence_reasoning": "..."
    }
  ]
}
```

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `hybrid_search_across_all` | Semantic + keyword search across all emails |
| `grep_across_all` | Regex search for exact patterns/names |
| `search_across_person` | Search within a specific person's emails |
| `get_random_across_person` | Random email samples from a person |
| `get_thread` | Read full thread content by ID |

## Configuration

- **`CHROMA_API_KEY`** / **`CHROMA_DATABASE`** — ChromaDB Cloud for email threads
- **`OPENAI_API_KEY`** — Embeddings for hybrid search
- **`ANTHROPIC_API_KEY`** — LLM calls for exploration and verification
