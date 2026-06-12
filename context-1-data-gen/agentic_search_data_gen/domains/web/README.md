# Web Domain

Generates multi-hop search tasks from the web. From seed topics, an agent explores pages via search and scraping, formulates a question with 3 supporting URLs, then optionally chains tasks across hops. Distractor pages are mined to increase difficulty.

## Quick start

Run the full pipeline (explore → verify → distract → verify distractors → index):

```bash
python -m agentic_search_data_gen.domains.web \
  --seeds agentic_search_data_gen/domains/web/seeds.txt \
  --output data/web/output \
  --collection my_collection
```

With extension rounds for multi-hop tasks:

```bash
python -m agentic_search_data_gen.domains.web \
  --seeds agentic_search_data_gen/domains/web/seeds.txt \
  --output data/web/output \
  --collection my_collection \
  --extension-rounds 1
```

Defaults: explore/distract/extend use `claude-sonnet-4-5`, verify uses `claude-opus-4-5`, 8 workers. Override any per-stage model or iteration limit via `--explore-model`, `--verify-max-retries`, etc.

Run `python -m agentic_search_data_gen.domains.web --help` for all options.

## Pipeline (per-stage)

### 1. Explore

Generate a task (clues, question, truth, supporting URLs) from a seed topic.

```bash
python -m agentic_search_data_gen.domains.web.explore \
  -s agentic_search_data_gen/domains/web/seeds.txt -o data/web/output -w 8 -i 20 -m claude-sonnet-4-5
```

**Input:** Text file with one seed topic per line.
**Output:** One JSON per seed in the output directory, containing `tasks[0]` with `clues`, `question`, `truth`, `supporting_items`, and `items_and_contents`.

### 2. Verify

Extract and verify quotes from supporting items. Checks that clue quotes appear in the clue text, item quotes appear in the page content, and at least one item contains the truth.

```bash
python -m agentic_search_data_gen.domains.web.verify \
  -i data/web/output -w 8 -m claude-opus-4-5
```

Adds `passed_verification` to each task.

### 3. Distract

Mine distractor pages that are topically related but don't contain the truth.

```bash
python -m agentic_search_data_gen.domains.web.distract \
  -i data/web/output -w 8 -n 15 -m claude-sonnet-4-5
```

Adds `distractors` to each task.

### 4. Verify distractors

Re-run verify with `--distractors` to filter out distractors that inadvertently contain the truth.

```bash
python -m agentic_search_data_gen.domains.web.verify \
  -i data/web/output --distractors -w 8 -m claude-opus-4-5
```

### 5. Extend (optional, repeat from step 3)

Chain a second hop by finding a new question whose answer connects to a previous supporting URL.

```bash
python -m agentic_search_data_gen.domains.web.extend \
  -i data/web/output -w 8 -n 20 -m claude-sonnet-4-5
```

Appends a new task at `level=1` with a `bridging_item` linking hops. After extending, re-run verify (step 2), distract (step 3), and verify distractors (step 4) for the new hop.

### 6. Index

Index web page chunks into ChromaDB with BM25 sparse + OpenAI dense embeddings.

```bash
python -m agentic_search_data_gen.domains.web.index \
  -i data/web/output -c <collection_name>
```

**Input:** Directory containing task JSON files from previous pipeline steps.
**Output:** `<input-dir>/index_output/url_to_id.json` and `<input-dir>/index_output/<timestamp>.json` stats.

## Output format

Each JSON file contains:

```json
{
  "tasks": [
    {
      "level": 0,
      "clues": "...",
      "question": "...",
      "truth": "...",
      "truth_type": "...",
      "supporting_items": [
        {"id": "https://...", "clue_quotes": [...], "item_quotes": [...], "contains_truth": true}
      ],
      "items_and_contents": {"https://...": "page text..."},
      "distractors": [...]
    }
  ],
  "surfaced_urls": [...],
  "visited_urls": [...]
}
```

## Configuration

- **`SERPER_API_KEY`** — Google search and page scraping via Serper
- **`JINA_API_KEY`** — Backup page fetcher via Jina
- **`OPENAI_API_KEY`** — Embeddings for long-page semantic search and indexing
- **`ANTHROPIC_API_KEY`** — LLM calls
- **`CHROMA_API_KEY`** — ChromaDB cloud indexing
- **`CHROMA_DATABASE`** — ChromaDB database name

Truth types: 17 categories (person, date, number, location, organization, etc.)
