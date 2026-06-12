"""Process SEC filings corpus: download, chunk, and index into ChromaDB."""
import os
import sys
import warnings

os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")

from typing import List
from edgar import *
import tiktoken
import shutil
import json
import re
from pathlib import Path
import chromadb
from chromadb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import Bm25EmbeddingFunction as ChromaBm25EF, OpenAIEmbeddingFunction
import time
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
from dotenv import load_dotenv
import logging
from openai import OpenAI
from fastembed.sparse.bm25 import Bm25
from chromadb.utils.sparse_embedding_utils import normalize_sparse_vector
import threading
from collections import deque

load_dotenv()

logging.getLogger("edgar").setLevel(logging.ERROR)

DEFINED_FORMS = ["10-K", "10-Q", "20-F", "8-K"]
FORMS_TO_IGNORE = ["144", "25-NSE", "FWP", "SD", "4"]


class RateLimiter:
    """Thread-safe rate limiter using sliding window."""

    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.timestamps = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            sleep_time = 0
            with self.lock:
                now = time.time()
                # Remove timestamps outside the window
                while self.timestamps and now - self.timestamps[0] >= self.time_window:
                    self.timestamps.popleft()

                if len(self.timestamps) < self.max_requests:
                    # We can proceed
                    self.timestamps.append(now)
                    return
                else:
                    # Calculate wait time and release lock before sleeping
                    sleep_time = self.time_window - (now - self.timestamps[0]) + 0.01

            # Sleep outside the lock so other threads can proceed
            if sleep_time > 0:
                time.sleep(sleep_time)


# Global rate limiter for SEC requests: 10 requests/second
SEC_RATE_LIMITER = RateLimiter(max_requests=10, time_window=1.0)


class CorpusProcessor:
    def __init__(self, collection_name: str = "sec_filings", identity_str: str = None):
        if identity_str is None:
            raise ValueError("identity_str is required (e.g. 'Name email@example.com')")

        cache_dir = Path.home() / ".edgar"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        set_identity(identity_str)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_DATABASE")
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.bm25_model = Bm25(model_name="Qdrant/bm25")

        schema = Schema()

        sparse_ef = ChromaBm25EF(query_config={'task': 'document'})
        schema.create_index(
            config=SparseVectorIndexConfig(
                source_key=K.DOCUMENT,
                embedding_function=sparse_ef,
                bm25=True
            ),
            key="bm25_vector"
        )

        embedding_function = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        schema.create_index(config=VectorIndexConfig(
            space="cosine",
            embedding_function=embedding_function
        ))

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            schema=schema
        )

    def get_token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def is_corrupted_text(self, text: str) -> bool:
        if not text or len(text) < 100:
            return False

        corruption_markers = ['begin 644', '<PDF>', 'begin-base64', '%PDF-']
        for marker in corruption_markers:
            if marker in text[:500]:
                return True

        sample = text[:1000].replace('\n', '').replace(' ', '')
        if len(sample) < 50:
            return False

        alnum_ratio = sum(c.isalnum() for c in sample) / len(sample)
        if alnum_ratio > 0.95:
            return True

        return False

    def is_tabular_chunk(self, text: str, whitespace_fraction_threshold: float = 0.35, multi_space_line_threshold: float = 0.7) -> bool:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False

        chars = [c for c in text if c not in '\n\r']
        if not chars:
            return False

        space_tab_count = sum(c in ' \t' for c in chars)
        whitespace_fraction = space_tab_count / len(chars)

        multi_space_lines = sum(1 for l in lines if re.search(r'[ \t]{2,}', l))
        multi_space_line_ratio = multi_space_lines / len(lines)

        return whitespace_fraction >= whitespace_fraction_threshold and multi_space_line_ratio >= multi_space_line_threshold

    def chunk_text(self, text: str, chunk_size: int = 512) -> tuple[list[str], dict]:
        stats = {"excluded": 0, "included": 0}

        tokens = self.encoding.encode(text)
        if len(tokens) < 50:
            stats["excluded"] = 1
            return None, stats

        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        chunks_to_return = []

        for chunk in chunks:
            chunk_text = self.encoding.decode(chunk)
            if self.is_tabular_chunk(chunk_text):
                stats["excluded"] += 1
            else:
                chunks_to_return.append(chunk_text)
                stats["included"] += 1

        return chunks_to_return if chunks_to_return else None, stats

    def process_filing(self, accession_no: str, filing: Filing) -> tuple[dict | None, dict]:
        filing_dict = filing.to_dict()
        company = filing_dict['company']
        form_type = filing_dict['form']
        filing_date = filing_dict['filing_date'].strftime('%Y-%m-%d')

        stats = {"corrupted_filings": 0, "excluded": 0, "included": 0}

        # Rate limit SEC request
        SEC_RATE_LIMITER.acquire()
        markdown_text = filing.markdown()

        if self.is_corrupted_text(markdown_text):
            stats["corrupted_filings"] = 1
            return None, stats

        metadata_str = f"Company: {company}\nFiling Date: {filing_date}\nForm: {form_type}"
        chunks_to_return = {}
        chunk_i = 0

        if form_type in DEFINED_FORMS:
            try:
                SEC_RATE_LIMITER.acquire()
                obj = filing.obj()
                items = obj.items

                for item in items:
                    item_text = obj[item]
                    chunks, chunk_stats = self.chunk_text(item_text)
                    stats["excluded"] += chunk_stats["excluded"]
                    stats["included"] += chunk_stats["included"]

                    if chunks:
                        for chunk in chunks:
                            item_metadata = metadata_str + f"\nItem: {item}\nChunk {chunk_i}"
                            chunk_id = f"{accession_no}_{chunk_i}"
                            complete_chunk = item_metadata + "\n---\n" + chunk
                            chunks_to_return[chunk_id] = complete_chunk
                            chunk_i += 1

                return chunks_to_return if chunks_to_return else None, stats
            except:
                pass

        chunks, chunk_stats = self.chunk_text(markdown_text)
        stats["excluded"] += chunk_stats["excluded"]
        stats["included"] += chunk_stats["included"]

        if chunks:
            for chunk in chunks:
                item_metadata = metadata_str + f"\nChunk {chunk_i}"
                chunk_id = f"{accession_no}_{chunk_i}"
                complete_chunk = item_metadata + "\n---\n" + chunk
                chunks_to_return[chunk_id] = complete_chunk
                chunk_i += 1

        return chunks_to_return if chunks_to_return else None, stats

    def openai_embed_in_batches(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Batch embeddings optimally for OpenAI limits:
        - 300k tokens per request
        - 2048 items per request
        - 10,000 requests per minute
        - 5,000,000 tokens per minute

        We batch by tokens (up to 200k to be safe) and items (up to 500).
        """
        MAX_TOKENS_PER_BATCH = 200_000  # Conservative limit under 300k
        MAX_ITEMS_PER_BATCH = 500  # Conservative limit under 2048

        all_embeddings = []
        current_batch = []
        current_tokens = 0
        batch_count = 0

        for text in texts:
            token_count = self.get_token_count(text)

            # Check if we need to flush current batch before adding this text
            would_exceed_tokens = current_tokens + token_count > MAX_TOKENS_PER_BATCH
            would_exceed_items = len(current_batch) >= MAX_ITEMS_PER_BATCH

            if current_batch and (would_exceed_tokens or would_exceed_items):
                batch_count += 1
                print(f"    OpenAI batch {batch_count}: {len(current_batch)} items, {current_tokens} tokens", flush=True)
                batch_embeddings = [
                    response.embedding
                    for response in self.openai_client.embeddings.create(model=model, input=current_batch).data
                ]
                all_embeddings.extend(batch_embeddings)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += token_count

        # Process remaining batch
        if current_batch:
            batch_count += 1
            print(f"    OpenAI batch {batch_count}: {len(current_batch)} items, {current_tokens} tokens", flush=True)
            batch_embeddings = [
                response.embedding
                for response in self.openai_client.embeddings.create(model=model, input=current_batch).data
            ]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def bm25_embed(self, texts: List[str], metadatas: List[dict]) -> List[dict]:
        sparse_embeddings = [
            normalize_sparse_vector(indices=vec.indices.tolist(), values=vec.values.tolist())
            for vec in self.bm25_model.embed(texts)
        ]
        return [{**meta, "bm25_vector": sparse} for meta, sparse in zip(metadatas, sparse_embeddings)]

    def index_chunks(self, chunks_to_save: dict, ticker: str) -> None:
        """
        Index chunks to Chroma with optimized batching:
        - Chroma max batch size: 300 items
        """
        CHROMA_BATCH_SIZE = 300
        MAX_DOC_SIZE_BYTES = 16000  # Chroma limit is 16384, use 16000 for safety

        to_add = {}
        skipped_oversized = 0
        for accession_no, filing_data in chunks_to_save.items():
            for i, (chunk_id, chunk_text) in enumerate(filing_data['chunks'].items()):
                if len(chunk_text.encode('utf-8')) > MAX_DOC_SIZE_BYTES:
                    skipped_oversized += 1
                    continue
                to_add[chunk_id] = {
                    "text": chunk_text,
                    "metadata": {
                        "source": accession_no,
                        "ticker": ticker,
                        "chunk_i": i,
                        "form_type": filing_data['form_type'],
                        "filing_date": filing_data['filing_date']
                    }
                }

        if skipped_oversized > 0:
            print(f"  [{ticker}] Skipped {skipped_oversized} oversized chunks (>{MAX_DOC_SIZE_BYTES} bytes)", flush=True)

        if not to_add:
            print(f"  [{ticker}] No chunks to index", flush=True)
            return

        ids = list(to_add.keys())
        documents = [to_add[id]['text'] for id in ids]
        metadatas = [to_add[id]['metadata'] for id in ids]

        print(f"  [{ticker}] Indexing {len(ids)} chunks...", flush=True)

        # Get embeddings (OpenAI batching handled internally)
        try:
            print(f"  [{ticker}] Getting OpenAI embeddings...", flush=True)
            dense_embeddings = self.openai_embed_in_batches(documents)
            print(f"  [{ticker}] Got {len(dense_embeddings)} embeddings", flush=True)
        except Exception as e:
            print(f"  [{ticker}] ERROR in OpenAI embeddings: {e}", flush=True)
            raise

        metadatas_with_sparse = self.bm25_embed(documents, metadatas)

        # Upload to Chroma in batches of 300
        total_batches = (len(ids) + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE
        for batch_num, i in enumerate(range(0, len(ids), CHROMA_BATCH_SIZE)):
            batch_ids = ids[i:i + CHROMA_BATCH_SIZE]
            batch_documents = documents[i:i + CHROMA_BATCH_SIZE]
            batch_embeddings = dense_embeddings[i:i + CHROMA_BATCH_SIZE]
            batch_metadatas = metadatas_with_sparse[i:i + CHROMA_BATCH_SIZE]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )
                    print(f"  [{ticker}] Chroma batch {batch_num+1}/{total_batches} uploaded ({len(batch_ids)} items)", flush=True)
                    break
                except Exception as e:
                    print(f"  [{ticker}] Chroma batch {batch_num+1} attempt {attempt+1} failed: {e}", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

    def process_company(self, ticker: str, output_dir: str, index: bool = True) -> tuple[str, bool, str, dict]:
        try:
            # Rate limit the initial company lookup
            SEC_RATE_LIMITER.acquire()
            company = Company(ticker)
            company_name = company.name
            company_industry = company.industry
            print(f"[{ticker}] Started: {company_name}", flush=True)

            SEC_RATE_LIMITER.acquire()
            recent_filings = company.get_filings(year=2025)
            filings_list = list(recent_filings)
            print(f"[{ticker}] Found {len(filings_list)} filings", flush=True)

            if len(filings_list) > 3000:
                print(f"[{ticker}] SKIPPED - too many filings ({len(filings_list)} > 3000)", flush=True)
                return (ticker, False, f"Too many filings: {len(filings_list)}", None)

            chunks_to_save = {}
            available_forms = {}
            company_stats = {"corrupted_filings": 0, "excluded": 0, "included": 0}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for filing in filings_list:
                    filing_dict = filing.to_dict()
                    form_type = filing_dict['form']
                    accession_no = filing.accession_no
                    filing_date = filing_dict['filing_date'].strftime('%Y-%m-%d')

                    if form_type in FORMS_TO_IGNORE:
                        continue

                    filing_data = {"form_type": form_type, "filing_date": filing_date}

                    # Retry logic for individual filings
                    max_retries = 3
                    chunks = None
                    filing_stats = {"corrupted_filings": 0, "excluded": 0, "included": 0}

                    for attempt in range(max_retries):
                        try:
                            chunks, filing_stats = self.process_filing(accession_no, filing)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)
                            # On final failure, continue with empty stats

                    company_stats["corrupted_filings"] += filing_stats["corrupted_filings"]
                    company_stats["excluded"] += filing_stats["excluded"]
                    company_stats["included"] += filing_stats["included"]

                    if chunks:
                        if form_type not in available_forms:
                            available_forms[form_type] = [accession_no]
                        else:
                            available_forms[form_type].append(accession_no)

                        filing_data["chunks"] = chunks
                        chunks_to_save[accession_no] = filing_data

            contains_key_form = "10-K" in available_forms.keys() or "20-F" in available_forms.keys()

            to_save = {
                "ticker": ticker,
                "company_name": company_name,
                "company_industry": company_industry,
                "contains_key_form": contains_key_form,
                "available_forms": available_forms
            }

            if index:
                self.index_chunks(chunks_to_save, ticker)
            else:
                to_save["chunks"] = chunks_to_save

            with open(f"{output_dir}/{ticker}.json", "w") as f:
                json.dump(to_save, f, indent=4)

            print(f"[{ticker}] COMPLETED - {company_stats['included']} chunks indexed", flush=True)
            return (ticker, True, None, company_stats)

        except Exception as e:
            print(f"[{ticker}] FAILED: {e}", flush=True)
            return (ticker, False, str(e), None)

    def process_batch(self, tickers: list[str], output_dir: str, index: bool = True, max_workers: int = 8) -> dict:
        console = Console()
        successful = []
        failed = []

        total_stats = {"corrupted_filings": 0, "chunks_excluded": 0, "chunks_included": 0}

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/summaries").mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        ) as progress:
            overall_task = progress.add_task(f"[bold blue]Processing {len(tickers)} companies", total=len(tickers))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_company, ticker, output_dir, index): ticker for ticker in tickers}

                for future in as_completed(futures):
                    ticker, success, error, stats = future.result()

                    if success:
                        successful.append(ticker)
                        if stats:
                            total_stats["corrupted_filings"] += stats["corrupted_filings"]
                            total_stats["chunks_excluded"] += stats["excluded"]
                            total_stats["chunks_included"] += stats["included"]
                    else:
                        failed.append({"ticker": ticker, "error": error})

                    progress.advance(overall_task)

        total_chunks_considered = total_stats["chunks_excluded"] + total_stats["chunks_included"]

        console.print()
        console.print(f"[bold green]Successful:[/bold green] {len(successful)}")
        console.print(f"[bold red]Failed:[/bold red] {len(failed)}")
        console.print(f"[bold cyan]Chunks Statistics:[/bold cyan]")
        console.print(f"  Total chunks considered: {total_chunks_considered}")
        console.print(f"  Valid chunks: {total_stats['chunks_included']}")
        console.print(f"  Excluded chunks: {total_stats['chunks_excluded']}")
        console.print(f"  Corrupted filings: {total_stats['corrupted_filings']}")

        if failed:
            console.print("\n[bold red]Failed tickers:[/bold red]")
            for item in failed:
                console.print(f"  {item['ticker']}: {item['error']}")

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        summary = {
            "total_tickers": len(tickers),
            "successful": successful,
            "failed": failed,
            "max_workers": max_workers,
            "index_enabled": index,
            "chunk_statistics": {
                "total_chunks_considered": total_chunks_considered,
                "included_chunks": total_stats["chunks_included"],
                "excluded_chunks": total_stats["chunks_excluded"],
                "corrupted_filings": total_stats["corrupted_filings"]
            }
        }

        summary_path = Path(output_dir) / "summaries" / f"summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        console.print(f"\n[bold cyan]Summary saved to:[/bold cyan] {summary_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Process SEC filings for a batch of companies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input text file containing ticker symbols (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Directory to save output files (default: data)"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Disable indexing to Chroma Cloud, only generate corpus locally"
    )
    parser.add_argument(
        "--collection",
        default="sec_filings",
        help="Collection name (default: sec_filings)"
    )
    parser.add_argument(
        "--identity",
        required=True,
        help="Identity string for SEC EDGAR API (e.g. 'Name email@example.com')"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    with open(input_path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        print("Error: No tickers found in input file")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_tickers = set()
    tickers_to_process = []

    for ticker in tickers:
        json_path = output_dir / f"{ticker}.json"
        if json_path.exists():
            existing_tickers.add(ticker)
        else:
            tickers_to_process.append(ticker)

    console = Console()
    console.print(f"\n[bold cyan]Starting corpus processing[/bold cyan]")
    console.print(f"  Input file: {args.input}")
    console.print(f"  Output directory: {args.output}")
    console.print(f"  Total tickers in input: {len(tickers)}")
    console.print(f"  Already processed: {len(existing_tickers)}")
    console.print(f"  Tickers to process: {len(tickers_to_process)}")
    console.print(f"  Max workers: {args.max_workers}")
    console.print(f"  Indexing: {'Disabled' if args.no_index else 'Enabled'}")
    console.print(f"  SEC rate limit: 10 req/s")
    console.print()

    if not tickers_to_process:
        console.print("[bold green]All tickers have already been processed![/bold green]")
        sys.exit(0)

    processor = CorpusProcessor(
        collection_name=args.collection,
        identity_str=args.identity
    )

    processor.process_batch(
        tickers=tickers_to_process,
        output_dir=args.output,
        index=not args.no_index,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
