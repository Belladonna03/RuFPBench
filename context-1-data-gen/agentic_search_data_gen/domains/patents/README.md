# Patents Domain

Generates search tasks from USPTO patent office actions. Extracts examiner rejections (claim-to-prior-art mappings) from non-final rejection documents, then generates evaluation tasks that require finding the relevant prior art given a rejected claim.

## Setup

Install patents dependencies:

```bash
uv sync --extra patents
```

No separate verification stage; examiner rejections serve as ground truth.

## Environment variables

- `USPTO_API_KEY` — USPTO patent API access
- `SEARCH_API_KEY` — Google Patents search API (SearchAPI)
- `DATALAB_API_KEY` — PDF processing (Datalab SDK)
- `ANTHROPIC_API_KEY` — LLM calls for extraction and generation
- `OPENAI_API_KEY` — Embeddings for indexing
- `CHROMA_API_KEY` — ChromaDB cloud access
- `CHROMA_DATABASE` — ChromaDB database name

## Quick start

Run the full pipeline in one command:

```bash
python -m agentic_search_data_gen.domains.patents \
  --seeds agentic_search_data_gen/domains/patents/seeds.txt --output data/patents/output --collection my_collection
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds/-s` | required | Path to seeds file (one application number per line) |
| `--output/-o` | required | Output directory |
| `--collection/-c` | required | ChromaDB collection name |
| `--concurrent` | 10 | Max concurrent requests for process stage |
| `--max-workers/-w` | 4 | Parallel workers for extract/generate stages |
| `--extract-model` | `claude-opus-4-5` | Anthropic model for extraction stage |
| `--generate-model` | `claude-opus-4-5` | Anthropic model for generation stage |
| `--embedding-model` | `text-embedding-3-small` | OpenAI embedding model for indexing stage |

**Models:** Extract and generate stages default to `claude-opus-4-5`; embeddings default to `text-embedding-3-small` via OpenAI. All are configurable via the CLI flags above. Individual stage entry points also accept `--model` (extract/generate) or `--embedding-model` (index).

## Pipeline

### 1. Process

Download and parse USPTO patent office actions into structured JSON.

```bash
python -m agentic_search_data_gen.domains.patents.process \
  -i data/patents/application_numbers.txt -o data/patents/output
```

**Input:** Text file with application numbers (one per line).
**Output:** JSON files with parsed sections (`CTNF` for non-final rejections, `CLM` for claims, `892` for references).

**Expected errors:** Some patents will fail during processing — this is normal. Not all applications have the required document set (CTNF, CLM, SPEC, ABST, 892), and some documents may fail to download or parse. Common errors logged to `errors.json`:

- `"Failed to extract all required documents (CTNF, CLM, SPEC, ABST, 892)"` — the patent doesn't have all five document types before the first non-final rejection date
- `"Failed to process and extract document contents (text/PDF/references)"` — a document download or PDF/DOCX parsing failed
- `"Application number already exists in dataset (found as reference)"` / `"...as similar patent"` — deduplication of patents seen via references or similar-patent links
- `"No CTNF (non-final rejection) document found before target date"` — the patent has no non-final rejection on file

The pipeline continues past these errors. Successfully processed patents are saved as individual JSON files; all errors are collected in `errors.json` in the output directory.

### 2. Extract

Extract structured rejection data from non-final rejection text. Parses claim element mappings, prior art references, and examiner reasoning.

```bash
python -m agentic_search_data_gen.domains.patents.extract \
  -i data/patents/output -w 4
```

Adds `extraction_result` to each JSON with parsed rejections including claim text, reasoning, and claim-element-to-prior-art mappings with citation locations.

### 3. Generate

Generate evaluation tasks from extracted rejections. Each task describes a search scenario where the goal is to find the prior art that maps to a rejected patent claim.

```bash
python -m agentic_search_data_gen.domains.patents.generate \
  -i data/patents/output -w 6
```

Adds `eval` to each extracted rejection item with `task` (the search task text) and `positive_docids` (the correct prior art document IDs).

### 4. Index

Index patent chunks into ChromaDB with BM25 sparse + OpenAI dense embeddings. Auto-discovers JSON files from the output directory.

```bash
python -m agentic_search_data_gen.domains.patents.index \
  -i data/patents/output -c <collection_name>
```

**Input:** Directory containing patent JSON files (auto-discovered, excluding `errors.json` and `index_output/`).
**Output:** `index_output/app_no_to_distractors.json` mapping application numbers to their distractor patents, plus timestamped stats JSON.

## Output format

```json
{
  "application_no": "17123456",
  "CTNF": {"text": "..."},
  "CLM": {"text": "..."},
  "892": {"references": {"US12345678": {"inventors": [...], "abstract": "..."}}},
  "extraction_result": {
    "num_extracted": 3,
    "extracted": [
      {
        "type": "103",
        "claim_number": "1",
        "claim_text": "...",
        "reasoning": "...",
        "claim_element_mappings": [...],
        "eval": {
          "task": "...",
          "positive_docids": ["US12345678"]
        }
      }
    ]
  }
}
```
