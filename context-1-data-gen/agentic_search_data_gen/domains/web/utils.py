from typing import Dict, Any, List, Optional, Tuple
import os
import re
import requests
import json
import uuid
import time
from dotenv import load_dotenv
import tiktoken
import chromadb
from openai import OpenAI as OpenAIClient

load_dotenv()

MAX_PAGE_TOKENS = 10000

# Connection pooling for HTTP requests
_session = None


def _get_session() -> requests.Session:
    """Get or create a shared requests session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure connection pool size for parallel requests
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


CHUNK_SIZE_TOKENS = 512
TOP_K_RESULTS = 10

# Initialize tiktoken encoder (cl100k_base is used by GPT-4 and similar models)
_tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

# Initialize OpenAI client for embeddings
_openai_client = None


def _get_openai_client() -> OpenAIClient:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using tiktoken."""
    return len(_tiktoken_encoder.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """
    Chunk text into segments of approximately chunk_size tokens.
    Tries to break at sentence boundaries when possible.
    """
    tokens = _tiktoken_encoder.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = _tiktoken_encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


MAX_EMBEDDING_TOKENS_PER_REQUEST = 200000  # OpenAI limit is 300k, use 200k for safety margin


def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI. Batches requests to stay under token limit."""
    client = _get_openai_client()

    # Batch texts to stay under token limit
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        text_tokens = count_tokens(text)
        if current_tokens + text_tokens > MAX_EMBEDDING_TOKENS_PER_REQUEST and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens

    if current_batch:
        batches.append(current_batch)

    # Get embeddings for each batch
    all_embeddings = []
    try:
        for batch in batches:
            response = client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []


def search_long_page(
    page_text: str,
    query: str,
    top_k: int = TOP_K_RESULTS
) -> Tuple[str, str]:
    """
    Create a temporary chroma collection from page chunks and search over it.
    Returns a tuple of (formatted_results, raw_chunks):
        - formatted_results: For display to the agent with result numbers and similarity scores
        - raw_chunks: Just the chunk text joined by double newlines (for saving)
    """
    # Chunk the text
    chunks = chunk_text(page_text, CHUNK_SIZE_TOKENS)

    if not chunks:
        return "No content to search.", ""

    # Get embeddings for chunks
    chunk_embeddings = get_embeddings(chunks)

    if not chunk_embeddings:
        return "Error: Could not generate embeddings for page content.", ""

    # Create a temporary local chroma collection
    chroma_client = chromadb.Client()
    collection_name = f"temp_page_{uuid.uuid4().hex[:8]}"
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add chunks to collection
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=chunk_embeddings
    )

    # Get query embedding and search
    query_embedding = get_embeddings([query])
    if not query_embedding:
        chroma_client.delete_collection(collection_name)
        return "Error: Could not generate embedding for query.", ""

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, len(chunks))
    )

    # Clean up the temporary collection
    chroma_client.delete_collection(collection_name)

    # Format results
    if not results["documents"] or not results["documents"][0]:
        return "No relevant content found.", ""

    raw_chunks = []
    formatted_results = []
    for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
        raw_chunks.append(doc)
        formatted_results.append(f"[Result {i+1}] (similarity: {1-distance:.3f})\n{doc}")

    return "\n\n---\n\n".join(formatted_results), "\n\n".join(raw_chunks)


LONG_PAGE_QUERY_MESSAGE = """The page you requested is too long ({token_count} tokens, max is {max_tokens}). To retrieve relevant information from this page, please provide a single search query that describes what you're looking for.

Output ONLY the search query, nothing else."""


def handle_long_page(
    client,
    url: str,
    page_content: str,
    input_messages: List[Dict[str, Any]],
    iteration: int = 0
) -> Tuple[str, bool, Optional[str], Optional[str]]:
    """
    Handle a long page by asking the agent for a search query and performing semantic search.

    Returns:
        Tuple of (output_text, was_long_page, search_query_used, content_for_save)
        - output_text: The formatted output shown to the agent
        - was_long_page: Whether the page exceeded MAX_PAGE_TOKENS
        - search_query_used: The search query used (if semantic search was performed)
        - content_for_save: The raw chunks joined by double newlines (for saving to urls_and_contents)
    """
    token_count = count_tokens(page_content)

    if token_count <= MAX_PAGE_TOKENS:
        return f"[Tool call #{iteration+1}] Contents of {url}:\n{page_content}", False, None, None

    # Page is too long - ask for a search query
    search_query = ask_agent_for_page_query(client, url, token_count, input_messages)

    truncation_prefix = f"[Page truncated from {token_count} tokens]\n"

    if search_query:
        # Perform semantic search over the page
        formatted_result, raw_chunks = search_long_page(page_content, search_query)
        output = f"[Tool call #{iteration+1}] Page at {url} was too long ({token_count} tokens). Searched with query '{search_query}'. Top {TOP_K_RESULTS} results:\n\n{formatted_result}"
        content_for_save = truncation_prefix + raw_chunks
        return output, True, search_query, content_for_save
    else:
        # Fallback: truncate the page
        truncated = _tiktoken_encoder.decode(_tiktoken_encoder.encode(page_content)[:MAX_PAGE_TOKENS])
        output = f"[Tool call #{iteration+1}] Contents of {url} (truncated from {token_count} tokens):\n{truncated}"
        content_for_save = truncation_prefix + truncated
        return output, True, None, content_for_save


def ask_agent_for_page_query(
    client,
    url: str,
    token_count: int,
    input_messages: List[Dict[str, Any]]
) -> str:
    """
    Ask the agent for a search query to use on a long page.
    This call does NOT include tools - we only want a text query back.
    """
    query_request_message = LONG_PAGE_QUERY_MESSAGE.format(
        token_count=token_count,
        max_tokens=MAX_PAGE_TOKENS
    )

    # Create a temporary message list, excluding any pending assistant tool_use message.
    # At this point, the last message might be an assistant message with tool_use blocks
    # that don't have corresponding tool_results yet. We must exclude it to avoid the
    # API error: "tool_use ids were found without tool_result blocks immediately after"
    temp_messages = input_messages.copy()

    if temp_messages and temp_messages[-1].get("role") == "assistant":
        content = temp_messages[-1].get("content", [])
        if isinstance(content, list):
            has_tool_use = any(
                (isinstance(item, dict) and item.get("type") == "tool_use")
                for item in content
            )
            if has_tool_use:
                temp_messages = temp_messages[:-1]

    temp_messages.append({
        "role": "user",
        "content": f"[System] For URL {url}: {query_request_message}"
    })

    # Make request WITHOUT tools to get just a text response
    response = client.messages.create(
        model="claude-sonnet-4-5",
        system="You are a helpful assistant. Respond with only the search query, no other text.",
        max_tokens=200,
        messages=temp_messages
        # No tools parameter - we want text only
    )

    # Extract the text response
    for item in response.content:
        if getattr(item, 'type', None) == 'text':
            return item.text.strip()

    return ""


TOOLS = [
    {
        "name": "get_page",
        "description": "Get the contents of a page from a given URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact url to get the contents of (i.e. https://www.cmu.edu/compbio/)"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "search",
        "description": "Search the web for a given query, returns the top results (along with their snippets and URLs)",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the web for"
                }
            },
            "required": ["query"]
        }
    }
]

def get_page(url: str, max_retries: int = 3) -> str:
    session = _get_session()
    payload = {
      "url": url
    }
    headers = {
      'X-API-KEY': os.getenv('SERPER_API_KEY', ''),
      'Content-Type': 'application/json'
    }
    last_error = None
    for attempt in range(max_retries):
        try:
            response = session.post("https://scrape.serper.dev", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'text' in data:
                return data['text']
            raise ValueError("Response missing 'text' field")
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            # Try jina backup before retrying serper
            text = get_page_jina_backup(url)
            if text is not None:
                return text
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Warning: Failed to fetch {url}: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

    return f"Error fetching page: {str(last_error)}"

def get_page_jina_backup(url: str) -> Optional[str]:
    """Backup page fetcher using Jina. Returns None on failure."""
    session = _get_session()
    full_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
        "X-Retain-Images": "none",
        "X-Remove-Selector": 'header, .class, #id, nav, footer, aside, .sidebar, .comments, .social'
    }

    try:
        response = session.get(full_url, headers=headers, timeout=30)
        response.raise_for_status()
        text = response.text
        marker = "Markdown Content:\n"
        if marker in text:
            return text.split(marker, 1)[1]
        return text
    except requests.RequestException as e:
        print(f"Warning: Jina backup also failed for {url}: {e}")
        return None


def search(query: str) -> List[Dict[str, Any]]:
    session = _get_session()
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.getenv('SERPER_API_KEY', ''),
        'Content-Type': 'application/json'
    }

    try:
        response = session.post(url, headers=headers, data=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "organic" not in data:
            return []

        results_cleaned = []
        for result in data["organic"]:
            if "amazon.com" in result.get("link", ""):
                continue
            # if "wikipedia.org" in result.get("link", ""):
            #     continue
            results_cleaned.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            })
        return results_cleaned
    except requests.RequestException as e:
        print(f"Search error: {str(e)}")
        return []


def format_search_results(search_results: List[Dict[str, Any]]) -> str:
    formatted_str = ""
    for result in search_results:
        formatted_str += f"Title: {result['title']}\n"
        formatted_str += f"Link: {result['link']}\n"
        formatted_str += f"Snippet: {result['snippet']}\n\n"
        formatted_str += "--------------------------------\n"
    return formatted_str


def truncate_long_page(content: str) -> str:
    """
    Truncate a long page to MAX_PAGE_TOKENS.
    Returns the truncated content with a header indicating truncation.
    If the page is not long, returns the original content.
    """
    token_count = count_tokens(content)
    if token_count <= MAX_PAGE_TOKENS:
        return content
    truncated = _tiktoken_encoder.decode(_tiktoken_encoder.encode(content)[:MAX_PAGE_TOKENS])
    return f"[Page truncated from {token_count} tokens]\n{truncated}"


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an item by converting 'url' key to 'id' for consistent JSON storage.
    This ensures all items use 'id' as the identifier key regardless of source.
    """
    normalized = item.copy()
    if 'url' in normalized:
        normalized['id'] = normalized.pop('url')
    return normalized


def denormalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Denormalize an item by converting 'id' key to 'url' for internal processing.
    This restores the 'url' key for use with web-related functions.
    """
    denormalized = item.copy()
    if 'id' in denormalized:
        denormalized['url'] = denormalized.pop('id')
    return denormalized
