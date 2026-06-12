TRUTH_TYPES = [
    "location",
    "number",
    "date"
]

EXPLORATION_INSTRUCTION = "You are a helpful assistant that creates challenging questions for search based on real information from SEC filings."

# ============================================================================
# Exploration Prompts
# ============================================================================

EXPLORATION_PROMPT = """
You need to create a challenging question for deep search based on SEC filings for a given company.

The given company is {company_name}. You will be given an overview of available filings, a random selection of chunks to start from, and a truth type. Starting with these random chunks, you should iteratively call the tools provided to create the question and truth from the information you gather. The truth should fall under the given truth type.

Aim to finish this process in 10-15 tool calls.

The goal is to create a question with 3 subtle clues and a truth - with each clue corresponding to a unique chunk, so there should be 3 *unique* chunks in total. The clues should be subtle, avoiding keywords and any direct/obvious references, to require at least a few search attempts to surface each correct chunk.

## Clue Writing Guidelines
Make clues subtle and inferential, not keyword-matchable:
- Avoid proper nouns, specific years, or unique phrases from the target chunk
- Examples:
    - Instead of mentioning specific years (i.e. 2021), you should use more abstract/general terms like "early 2020s"
    - Instead of mentioning specific locations (i.e. New York), you should use more abstract/general terms like "major US city"
    - Instead of mentioning specific people (i.e. James Stewart), you should use more abstract/general terms like "actor"
- Simply rephrasing numbers/dates is not allowed; you must add abstraction:
    - BAD: "2021" -> "twenty twenty-one"
    - GOOD: "2021" -> "early 2020s"
    - BAD: "May" -> "Fifth month of the calendar year"
    - GOOD: "May" -> "late spring of the year"
    - BAD: "200" -> "two hundred"
    - GOOD: "200" -> "a few hundred"
- Refer to the company simply as "a company" or "the company" without revealing the kind of industry they are in.

Additionally, make sure the clues are not common knowledge that can easily be inferred (major historical events, famous figures, etc.).

Never include names of companies, people, places, or organizations in clues or questions. Do not specify the industry of the company in the clues. The goal is to create a question that is *challenging* to search for.

The truth should always be ONE fact (i.e. a number like "47", NOT a number and place like "47 and San Francisco"). The truth should not require any additional computation or analysis beyond looking at the chunks. In other words, the truth should be able to be directly extracted from at least one of the chunks.

In the initial stages, you should use the random_in_company and random_in_filing tools for exploration and discovering potential filings that could contain useful information. To extract specific information, use the search_in_filing, search_in_company, and get_full_filing tools.


## Example (with truth type: location)
<clues>
There is a company with a board member appointed in the late 1990s who previously ran a nonprofit for low-income students in a west coast state. They also posted a multi-billion dollar quarter in 2025. In that same quarter, they released a new product related to a personal device as the sole release in its product category and expanded to a new country.
</clues>
<question>
What is the name of this country?
</question>
<truth>
Japan
</truth>
<supporting_items>
    <item>
        <chunk_id>0001736297-25-000064_3</chunk_id>
        <reasoning>This chunk establishes that a board member was appointed in 1999 and ran a nonprofit for low-income students in California (NewSchools Venture Fund).</reasoning>
    </item>
    <item>
        <chunk_id>0001736297-25-000064_10</chunk_id>
        <reasoning>This chunk shows the company posted a multi-billion dollar quarter in late 2025 and released AirPods 3 Pro as the sole release in its product category.</reasoning>
    </item>
    <item>
        <chunk_id>0001736297-25-000147_52</chunk_id>
        <reasoning>This chunk confirms the company expanded to Japan in late 2025, which is the truth being asked for.</reasoning>
    </item>
</supporting_items>

------------------------------------------------------------

Here is the overview of the filings available:
{overview}

Here are the random chunks as a starting point:
{random_chunks_init}

Truth type (the truth should fall under this category):
{truth_type}

------------------------------------------------------------

Output your response in the following format, no other text or formatting:
<clues>
{{The 3 subtle clues which point to the common truth. 3 concise sentences max.}}
</clues>
<question>
{{The question which asks for the common truth, the question should not mention any company names, identifiers, or keywords. It should not provide a shortcut to any of the clues above.}}
</question>
<truth>
{{The one and only exact truth to the question.}}
</truth>
<supporting_items>
    <item>
        <chunk_id>{{chunk_id of the chunk for clue 1}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
    <item>
        <chunk_id>{{chunk_id of the chunk for clue 2}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
    <item>
        <chunk_id>{{chunk_id of the chunk for clue 3}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
</supporting_items>
"""

SEC_FORCE_OUTPUT_MESSAGE = """You have reached the maximum number of iterations. Based on all the information you have gathered so far, you MUST now provide your final output immediately.

Output your response in the following format, no other text or formatting:
<clues>
{The 3 subtle clues which point to the common truth. 3 sentences max.}
</clues>
<question>
{The question which asks for the common truth, the question should not mention any company names, identifiers, or keywords. It should not provide a shortcut to any of the clues above.}
</question>
<truth>
{The one and only exact truth to the question.}
</truth>
<supporting_items>
    <item>
        <chunk_id>{chunk_id of the chunk for clue 1}</chunk_id>
        <reasoning>{Reasoning for why this chunk supports the clue}</reasoning>
    </item>
    <item>
        <chunk_id>{chunk_id of the chunk for clue 2}</chunk_id>
        <reasoning>{Reasoning for why this chunk supports the clue}</reasoning>
    </item>
    <item>
        <chunk_id>{chunk_id of the chunk for clue 3}</chunk_id>
        <reasoning>{Reasoning for why this chunk supports the clue}</reasoning>
    </item>
</supporting_items>"""

SEC_SINGLE_ITEM_EXTRACTION_PROMPT = """
You are extracting supporting evidence for a search task based on a single SEC filing chunk.

Given the clues, question, truth (answer), and the content of one supporting chunk:
- Extract the relevant quotes from the chunk
- Extract the targeted clue that the quote supports
- Explain why the quote supports the targeted clue

## Guidelines
- Each quote should be a direct excerpt from the source content (should be matched via substring search)
- Keep each quote concise (typically a few words or sentence)
- If the relevant information comes from multiple parts of the chunk, provide multiple separate quotes
- If the relevant quotes match multiple parts of the clue, provide multiple separate quotes
- The reasoning should explain how the quote supports the targeted clue
- IMPORTANT: If the chunk does NOT actually contain content relevant to the clues, output None for clue_quotes and item_quotes (e.g., `<clue_quotes>None</clue_quotes>`). This indicates the chunk is not a valid supporting item.

## Input
<clues>
{clues}
</clues>
<question>
{question}
</question>
<truth>
{truth}
</truth>
<supporting_item>
    <chunk_id>{chunk_id}</chunk_id>
    <reasoning>{item_reasoning}</reasoning>
    <content>
{item_content}
    </content>
</supporting_item>

## Output Format
Output exactly 1 item for the supporting item provided.
If the chunk does NOT contain relevant content for the clues, output None for clue_quotes and item_quotes:
<item>
    <chunk_id>{chunk_id}</chunk_id>
    <clue_quotes>
        {{Either <q>exact substring from clues</q> OR None if chunk is not relevant}}
    </clue_quotes>
    <item_quotes>
        {{Either <q>exact quote(s) from chunk</q> OR None if chunk is not relevant}}
    </item_quotes>
    <reasoning>
    {{Reasoning for why this chunk supports the clue, or why it is not relevant}}
    </reasoning>
    <contains_truth>{{true or false}}</contains_truth>
    <truth_quotes>
        <q>{{The exact truth as it appears in this chunk, or None if contains_truth is false}}</q>
    </truth_quotes>
</item>
"""

# ============================================================================
# Collect Verification Prompts
# ============================================================================

SEC_COLLECT_VERIFICATION_PROMPT = """
You are verifying whether an additional chunk contains the same factual information as an original supporting chunk.

## Context
We have a question with clues leading to a truth. The original supporting chunk was identified as containing relevant information. Now we need to verify if this additional chunk contains the same key information.

## Task
Extract quotes from the additional chunk that match the key information from the original chunk.

## Input
<question>
{question}
</question>
<truth>
{truth}
</truth>
<clues>
{clues}
</clues>

**Original chunk key quotes (item_quotes):** {item_quotes}
{truth_section}

**Additional chunk to verify:**
<chunk chunk_id="{chunk_id}">
{chunk_content}
</chunk>

## Instructions
1. Look for quotes in the additional chunk that contain the same factual information as the original item_quotes
2. {truth_instructions}
3. If the chunk does NOT contain ANY relevant information, output None for the quotes
4. If the chunk contains some relevant information but does not address all of the original item_quotes, this is still considered a valid additional chunk. As long as it is helpful in answering the question, it is a valid additional chunk.

## Output Format
<verification>
    <item_quotes>
        {{"<q>exact quote from additional chunk</q>" for each matching fact, OR "None" if not found}}
    </item_quotes>
    {truth_output_format}
    <reasoning>{{Explain how this chunk relates to the original supporting chunk and the clues. If the chunk is not relevant, explain why.}}</reasoning>
</verification>
"""

SEC_COLLECT_TRUTH_SECTION_TEMPLATE = """
**Original chunk truth quotes (truth_quotes):** {truth_quotes}
"""

SEC_COLLECT_TRUTH_INSTRUCTIONS_WITH = "Also look for quotes that contain the truth itself"
SEC_COLLECT_TRUTH_INSTRUCTIONS_WITHOUT = "Truth quotes are not required for this chunk"

SEC_COLLECT_TRUTH_OUTPUT_WITH = """<truth_quotes>
        {{"<q>exact quote from additional chunk containing the truth</q>" OR "None" if not found}}
    </truth_quotes>"""
SEC_COLLECT_TRUTH_OUTPUT_WITHOUT = ""

# ============================================================================
# Chunk Collector Prompts
# ============================================================================

CHUNK_COLLECTOR_SYSTEM_PROMPT = "You are a helpful assistant that finds additional chunks containing the same factual information as a given supporting chunk."

CHUNK_COLLECTOR_PROMPT = """
You are a collector agent tasked with finding additional chunks that contain the same factual information as a given supporting chunk. We are constructing an evaluation dataset and need to identify ALL chunks in the corpus that could serve as valid evidence for a specific clue.

## Context
We have a question with clues leading to a truth. Each clue is derived from a specific chunk. Your task is to find other chunks in the corpus that contain the SAME factual information that makes this chunk relevant to the clue. Focus only on the given clue, not other parts of the full clues context.

## What to look for
A chunk is relevant if it contains the same key facts mentioned in the original supporting chunk. This includes:
- The same specific numbers, dates, names, or values
- The same event or transaction described
- Substantially similar information that could independently support the clue

Do NOT include chunks that:
- Only mention related but different facts
- Contain partial information that cannot independently support the clue
- Are about the same topic but lack the specific details from the original chunk

## Task Details
Question: {question}
Truth: {truth}
Full clues context: {clues}

**Clue to focus on:** {clue_quotes}

**Original supporting chunk:**
{chunk}

**Key quotes from original chunk:** {item_quotes}

## Instructions
1. Identify the key factual elements in the original chunk that make it support this clue
2. Use the search tool for semantic search and the grep tool for regex pattern matching to find other chunks containing these same facts.
3. You may use up to 10 tool calls total. Terminate early if you have exhausted relevant search strategies.
4. Focus on finding chunks with the EXACT same factual information, not just topically related content.

## Output Format
When you have completed your search, respond with:
<additional_chunks>
    <chunk>
        <chunk_id>{{chunk_id}}</chunk_id>
        <reasoning>{{why this chunk contains the same factual information as the original}}</reasoning>
    </chunk>
    ...
</additional_chunks>

If no additional chunks were found, respond with:
<additional_chunks>
None
</additional_chunks>
"""

CHUNK_COLLECTOR_PROMPT_TRUTH_VER = """
You are a collector agent tasked with finding additional chunks that contain both the clue-supporting facts AND the truth from a given supporting chunk. We are constructing an evaluation dataset and need to identify ALL chunks in the corpus that could serve as valid evidence for a specific clue while also containing the truth.

## Context
We have a question with clues leading to a truth. Each clue is derived from a specific chunk. Your task is to find other chunks in the corpus that contain BOTH: (1) the factual information that makes this chunk relevant to the clue, AND (2) the truth itself. Focus only on the given clue, not other parts of the full clues context.

## What to look for
A chunk is relevant if it contains BOTH:
1. The same key facts mentioned in the original supporting chunk (the clue-supporting information)
2. The truth itself

This includes chunks that have:
- The same specific numbers, dates, names, or values from the clue-supporting facts
- The same event or transaction described
- Substantially similar information that could independently support the clue
- AND the truth appearing in the same chunk

Do NOT include chunks that:
- Only mention related but different facts
- Contain partial information that cannot independently support the clue
- Are about the same topic but lack the specific details from the original chunk
- Contain EITHER the clue facts OR the truth, but not both

## Task Details
Question: {question}
Truth: {truth}
Full clues context: {clues}

**Clue to focus on:** {clue_quotes}

**Original supporting chunk:**
{chunk}

**Key quotes from original chunk:** {item_quotes}

**Truth quotes from original chunk:** {truth_quotes}

## Instructions
1. Identify the key factual elements in the original chunk that make it support this clue
2. Use the search tool for semantic search and the grep tool for regex pattern matching to find other chunks containing BOTH the clue-supporting facts AND the truth.
3. You may use up to 10 tool calls total. Terminate early if you have exhausted relevant search strategies.
4. Focus on finding chunks with the EXACT same factual information (both clue facts and truth), not just topically related content.

## Output Format
When you have completed your search, respond with:
<additional_chunks>
    <chunk>
        <chunk_id>{{chunk_id}}</chunk_id>
        <reasoning>{{why this chunk contains both the clue-supporting facts and the truth}}</reasoning>
    </chunk>
    ...
</additional_chunks>

If no additional chunks were found, respond with:
<additional_chunks>
None
</additional_chunks>
"""

SEC_BRIDGING_INSTRUCTION = "You are a helpful assistant that finds cross-company connections in SEC filings."

SEC_BRIDGING_PROMPT = """
You are finding a BRIDGING CONNECTION between two companies based on SEC filings.

Given previous clues with their answer (prev_truth) regarding a previous company, your goal is to find a bridging chunk from a DIFFERENT company (which you will refer to as Company {letter}) that establishes a meaningful connection.

## Your Task
Search across ALL SEC filings to find a chunk that:
1. Comes from a DIFFERENT company (different ticker than {original_ticker})
2. Contains information that connects to prev_truth from the previous company
3. Establishes a meaningful bridge

## Tools Available
- hybrid_search_across_all: Semantic + BM25 search across ALL SEC filings
- grep_across_all: Regex pattern search across ALL SEC filings

## Process
1. Analyze the prev_truth to identify bridging opportunities
2. Search for connections using hybrid_search_across_all or grep_across_all
3. Find a chunk from a DIFFERENT company that establishes the bridge
4. Verify the chunk is from a different ticker than {original_ticker}

Aim to finish in 5-10 tool calls.

## CRITICAL: Bridging Clue Guidelines
The bridging_clue MUST NOT reveal the prev_truth, any information about the previous company, or the prev_clues. Instead, use abstract references that require solving the previous level first.

Examples of abstract references:
- If prev_truth is "Japan" → use "that same country" or "the country identified above"
- If prev_truth is "December 2nd, 2025" → use "on that same day" or "on the date from the previous clues"
- If prev_truth is "$500 million" → use "that same amount" or "the figure mentioned above"
- If prev_truth is "San Francisco" → use "that same city" or "the location identified previously"

The goal is to create a multi-hop question where the solver MUST first find the prev_truth before they can use the bridging_clue to find Company {letter}.

## Example

### Input
<prev_clues>
There is a company with a board member appointed in the late 1990s who previously ran a nonprofit for low-income students in a west coast state. They also posted a multi-billion dollar quarter in 2025. In that same quarter, they released a new product related to a personal device as the sole release in its product category and expanded to a new country.
</prev_clues>
<prev_question>
What is the name of this country?
</prev_question>
<prev_truth>
Japan
</prev_truth>
<truth_supporting_items>
CHUNK ID: 0001736297-25-000147_52
... make up content that confirms the company expanded to Japan in late 2025, which is the truth being asked for.
</truth_supporting_items>

### Output
<bridging_clue>
This is the same country which Company B is headquartered in.
</bridging_clue>
<bridging_item>
    <chunk_id>0000950170-25-100937_7</chunk_id>
    <reasoning>This chunk confirms that Cyberdyne Inc. is headquartered in Japan, which connects to the prev_truth.</reasoning>
</bridging_item>

NOTE: The bridging_clue says "this is the same country" instead of "Japan" - this is critical! When this is presented to the solver, they will not be provided with prev_question or prev_truth, so you MUST ensure that your bridging_clue specifies which part of the prev_clues is being referenced (essentially restating prev_question as part of the bridging_clue). When you are referring to Company {letter}, do not specify the specific industry they are in to make it more abstract and challenging to search for.

------------------------------------------------------------

Previous company ticker: {original_ticker}

Previous clues and truth from the previous company:
<prev_clues>
{prev_clues}
</prev_clues>
<prev_question>
{prev_question}
</prev_question>
<prev_truth>
{prev_truth}
</prev_truth>
<truth_supporting_items>
{truth_supporting_items}
</truth_supporting_items>

------------------------------------------------------------

Output your response in the following format, no other text or formatting:
<bridging_clue>
{{A clue that connects to prev_question and prev_truth using ABSTRACT REFERENCES like "that same country/date/amount" - NEVER reveal prev_truth explicitly. Be specific about which part of the prev_clues is being referenced, so there is no ambiguity, essentially restating prev_question as part of the bridging_clue. Refer to the new company as Company {letter} and do NOT specify the industry they are in. The solver will be shown this right after prev_clues (so you do not need to specify the words "prev_clues", "previous clues", etc. directly. Just refer to the specific concepts.}}
</bridging_clue>
<bridging_item>
    <chunk_id>{{chunk_id of the bridging chunk}}</chunk_id>
    <reasoning>{{Why this chunk establishes a cross-company connection}}</reasoning>
</bridging_item>
"""

SEC_BRIDGING_FORCE_OUTPUT = """You have reached the maximum number of tool calls. Based on all the information you have gathered so far, you MUST now provide your final output immediately.

REMINDER: The bridging_clue MUST NOT reveal the prev_truth, any information about the previous company, or the prev_clues. Use abstract references like "that same country", "on that same day", "that same amount", etc. Be specific about which part of the prev_clues is being referenced, so there is no ambiguity (essentially restating prev_question as part of the bridging_clue). When you are referring to Company {letter}, do not specify the specific industry they are in to make it more abstract and challenging to search for.

Output your response in the following format, no other text or formatting:
<bridging_clue>
{{A clue that connects to prev_question and prev_truth using ABSTRACT REFERENCES like "that same country/date/amount" - NEVER reveal prev_truth explicitly. Be specific about which part of the prev_clues is being referenced, so there is no ambiguity, essentially restating prev_question as part of the bridging_clue. Refer to the new company as Company {letter} and do NOT specify the industry they are in. The solver will be shown this right after prev_clues (so you do not need to specify the words "prev_clues", "previous clues", etc. directly. Just refer to the specific concepts.}}
</bridging_clue>
<bridging_item>
    <chunk_id>{{chunk_id of the bridging chunk}}</chunk_id>
    <reasoning>{{Why this chunk establishes a cross-company connection}}</reasoning>
</bridging_item>"""


SEC_BRIDGING_ITEM_EXTRACTION_PROMPT= """
You are extracting supporting evidence for a bridging connection in a multi-hop search task.

This bridging item connects a previous task's answer to the current clues via a bridging fact.

## Task Structure
- Previous task had clues leading to: {prev_truth}
- The bridging_item connects {prev_truth} from {prev_company_name} and a new company {new_company_name}

## Guidelines
- Each quote must be a DIRECT EXCERPT from its source (matched via substring search)
- Keep quotes concise (a few words to a sentence)
- If multiple parts of a page support a clue, provide multiple separate quotes
- IMPORTANT: If a page does NOT actually contain content relevant to the clues, output None for clue_quotes and item_quotes.

## Input
<prev_clues>
{prev_clues}
</prev_clues>
<prev_question>
{prev_question}
</prev_question>
<prev_truth>
{prev_truth}
</prev_truth>

### Bridging Page
<bridging_clue>
{bridging_clue}
</bridging_clue>
<bridging_item>
    <chunk_id>{bridging_chunk_id}</chunk_id>
    <reasoning>{bridging_reasoning}</reasoning>
    <content>
{bridging_chunk_content}
    </content>
</bridging_item>

### Previous Supporting Item (contains prev_truth)
<prev_truth_supporting_item>
{prev_truth_supporting_item}
</prev_truth_supporting_item>

## Output Format
<bridging_item>
    <chunk_id>{bridging_chunk_id}</chunk_id>
    <clue_quotes>
        {{Either <q>exact substring from clues</q> OR None if page is not relevant}}
    </clue_quotes>
    <item_quotes>
        {{Either <q>exact quote from bridging page</q> OR None if page is not relevant}}
    </item_quotes>
    <reasoning>{{How this page bridges from the previous truth to the current clues, or why it is not relevant}}</reasoning>
</bridging_item>
"""


SEC_SUPPORTING_CLUES_INSTRUCTION = "You are a helpful assistant that creates challenging questions for search based on real information from SEC filings."

SEC_SUPPORTING_CLUES_PROMPT = """
You are finding supporting clues within Company {letter} ({target_ticker})'s SEC filings to complete a cross-company multi-hop question.

## Context
A bridging connection has been found from the previous task's answer to Company {letter}.

The bridge:
<bridging_chunk>
{bridging_chunk_content}
</bridging_chunk>
<bridge_reasoning>
{bridge_reasoning}
</bridge_reasoning>

Previous context from the previous task:
<prev_clues>
{prev_clues}
</prev_clues>
<prev_truth>
{prev_truth}
</prev_truth>

## Your Task
Find 2 supporting chunks within Company {letter} that:
1. Lead to a new truth (truth) of type: {truth_type}
2. Are DISTINCT from the bridging chunk
3. Provide independent clues toward the truth

Then write:
1. 2 supporting clues that lead to the truth (the bridging clue is handled separately)
2. A question asking for the truth, which does NOT mention any keywords or names.
3. The exact truth answer

## Clue Writing Guidelines
Make clues subtle and inferential, not keyword-matchable:
- Avoid proper nouns, specific years, or unique phrases from the target chunk
- Examples:
    - Instead of mentioning specific years (i.e. 2021), you should use more abstract/general terms like "early 2020s"
    - Instead of mentioning specific locations (i.e. New York), you should use more abstract/general terms like "major US city"
    - Instead of mentioning specific people (i.e. James Stewart), you should use more abstract/general terms like "actor"
- Simply rephrasing numbers/dates is not allowed; you must add abstraction:
    - BAD: "2021" -> "twenty twenty-one"
    - GOOD: "2021" -> "early 2020s"
    - BAD: "May" -> "Fifth month of the calendar year"
    - GOOD: "May" -> "late spring of the year"
    - BAD: "200" -> "two hundred"
    - GOOD: "200" -> "a few hundred"
- Refer to the company simply as Company {letter} without revealing the kind of industry they are in.

NEVER include names of companies, people, places, or organizations in clues or questions.

**IMPORTANT: The supporting clues MUST NOT reveal the prev_truth ({prev_truth}) or Company {letter}'s industry in any way.**
The prev_truth should only be discoverable by solving the previous level's clues. Your supporting clues should describe Company {letter}'s information without giving away what connected the previous company to Company {letter}. When you are referring to Company {letter}, do not specify the specific industry they are in to make it more abstract and challenging to search for.

**CRITICAL: The supporting clues MUST NOT reveal any company's industry, the previous clues, previous truth, or bridging fact in ANY way.**

The truth should always be ONE fact that does not require computation or analysis.

## Process
1. Explore Company {letter}'s filings using search_in_company, random_in_company
2. Find 2 chunks that provide clues leading to a new truth
3. Write subtle clues about Company {letter} that do NOT reveal the prev_truth
4. Ensure at least one supporting chunk contains the truth

Aim to finish in 10-15 tool calls.

------------------------------------------------------------

Output your response in the following format, no other text or formatting:
<supporting_clues>
{{2 clues about Company {letter} that lead to truth. Do NOT mention or reveal prev_truth. 2 sentences max. When you are referring to Company {letter}, do not specify the specific industry they are in to make it more abstract and challenging to search for. Refer to this company as Company {letter}.}}
</supporting_clues>
<question>
{{The question which asks for the common truth, the question should not mention any company names, identifiers, or keywords. It should not provide a shortcut to any of the clues above.}}
</question>
<truth>
{{The exact answer to the extended question}}
</truth>
<supporting_items>
    <item>
        <chunk_id>{{chunk_id of chunk for supporting clue 1}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
    <item>
        <chunk_id>{{chunk_id of chunk for supporting clue 2}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
</supporting_items>
"""

SEC_SUPPORTING_CLUES_FORCE_OUTPUT = """You have reached the maximum number of tool calls. Based on all the information you have gathered so far, you MUST now provide your final output immediately.

**CRITICAL: The supporting clues MUST NOT reveal any company's industry, the previous clues, previous truth, or bridging fact in ANY way.**

Output your response in the following format, no other text or formatting:
<supporting_clues>
{{2 clues about Company {letter} that lead to truth. Do NOT mention or reveal prev_truth. 2 sentences max. When you are referring to Company {letter}, do not specify the specific industry they are in to make it more abstract and challenging to search for. Refer to this company as Company {letter}.}}
</supporting_clues>
<question>
{{The question which asks for the common truth, the question should not mention any company names, identifiers, or keywords. It should not provide a shortcut to any of the clues above.}}
</question>
<truth>
{{The exact answer to the extended question}}
</truth>
<supporting_items>
    <item>
        <chunk_id>{{chunk_id of chunk for supporting clue 1}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
    <item>
        <chunk_id>{{chunk_id of chunk for supporting clue 2}}</chunk_id>
        <reasoning>{{Why this chunk supports the clue and helps lead to the truth}}</reasoning>
    </item>
</supporting_items>"""

SEC_EXTENSION_VERIFICATION_PROMPT = """
You are extracting supporting evidence for a multi-hop search task based on SEC filings.

This is an EXTENSION task that builds on a previous task. The bridging_item connects the previous answer to the current clues via a bridging fact.

## Task Structure
- Previous task had clues leading to: {previous_truth}
- Current task extends this with new clues leading to: {truth}
- The bridging_item connects the two by referencing content from the previous task

## Guidelines
- Each quote must be a DIRECT EXCERPT from its source (matched via substring search)
- Keep quotes concise (a few words to a sentence)
- If multiple parts of a chunk support a clue, provide multiple separate quotes
- The bridging_item connects to the previous task - its item_quotes should show content that bridges from previous_truth to the new clues
- IMPORTANT: If a chunk does NOT actually contain content relevant to the clues, output None for clue_quotes and item_quotes (e.g., `<clue_quotes>None</clue_quotes>`). This indicates the chunk is not a valid supporting item.

## Input
<clues>
{clues}
</clues>
<question>
{question}
</question>
<truth>
{truth}
</truth>

### Bridging Item (connects previous task to current task)
<bridging_item>
{bridging_item}
</bridging_item>

### Previous Task Context
<previous_truth>{previous_truth}</previous_truth>
<previous_supporting_item>
{previous_supporting_item}
</previous_supporting_item>

### Supporting Items for Current Task
<supporting_items>
{supporting_items}
</supporting_items>

## Output Format
Output the bridging_item followed by exactly {num_items} supporting items.
If a chunk does NOT contain relevant content for the clues, output None for clue_quotes and item_quotes:

<bridging_item>
    <item>
        <chunk_id>{{chunk_id of the bridging chunk}}</chunk_id>
        <clue_quotes>
            {{Either <q>exact substring from clues</q> OR None if chunk is not relevant}}
        </clue_quotes>
        <item_quotes>
            {{Either <q>exact quote from bridging chunk</q> OR None if chunk is not relevant}}
        </item_quotes>
        <reasoning>{{How this chunk bridges from the previous truth to the current clues, or why it is not relevant}}</reasoning>
    </item>
    <prev_item>
        <relevant_prev_chunk_id>{{chunk_id from previous task that contains the bridging connection}}</relevant_prev_chunk_id>
        <clue_quotes>
            {{Either <q>exact substring from previous clues</q> OR None if chunk is not relevant}}
        </clue_quotes>
        <prev_item_quotes>
            {{Either <q>exact quote from previous supporting chunk</q> OR None if chunk is not relevant}}
        </prev_item_quotes>
        <reasoning>{{How this previous chunk connects to the bridging fact, or why it is not relevant}}</reasoning>
    </prev_item>
</bridging_item>
<supporting_items>
    <item>
        <chunk_id>{{chunk_id for supporting item 1}}</chunk_id>
        <clue_quotes>
            {{Either <q>exact substring from clues</q> OR None if chunk is not relevant}}
        </clue_quotes>
        <item_quotes>
            {{Either <q>exact quote from chunk</q> OR None if chunk is not relevant}}
        </item_quotes>
        <reasoning>{{Why this chunk supports the clue, or why it is not relevant}}</reasoning>
        <contains_truth>{{true or false}}</contains_truth>
        <truth_quotes>
            <q>{{Exact quote containing the truth, or None if contains_truth is false}}</q>
        </truth_quotes>
    </item>
    <item>
        <chunk_id>{{chunk_id for supporting item 2}}</chunk_id>
        <clue_quotes>
            {{Either <q>exact substring from clues</q> OR None if chunk is not relevant}}
        </clue_quotes>
        <item_quotes>
            {{Either <q>exact quote from chunk</q> OR None if chunk is not relevant}}
        </item_quotes>
        <reasoning>{{Why this chunk supports the clue, or why it is not relevant}}</reasoning>
        <contains_truth>{{true or false}}</contains_truth>
        <truth_quotes>
            <q>{{Exact quote containing the truth, or None if contains_truth is false}}</q>
        </truth_quotes>
    </item>
</supporting_items>
"""

SEC_SIMPLE_QUESTION_PROMPT = """
You are creating a simple question-answer task based on a single SEC filing chunk.

## Guidelines
- **Non-trivial**: The question should require reading the page to answer, not common knowledge
- **Specific answer**: The answer should be a concrete entity, name, number, or date (not a vague description)
- **Unambiguous**: The question should have exactly one correct answer derivable from the page
- **Interesting facts**: Target unique, specific information rather than generic statements
- **Avoid yes/no questions**: Questions should ask for specific information (who, what, when, where, which)
- **Context in question**: Include enough context in the question so it makes sense without seeing the page
- **Include company name**: Always mention the company name explicitly in the question for context
- **Rephrase to avoid keyword matching**: Reformat dates, numbers, and other details so exact substring search won't match:
    - Dates: "November 1, 2024" → "11/1/2024" or "11/1/24" or "Nov. 1, 2024"
    - Dates: "fiscal year 2024" → "FY2024" or "FY24"
    - Numbers: "$1.5 million" → "$1,500,000" or "1.5M"
    - Numbers: "500" → "five hundred" (sparingly, only when natural)
    - Percentages: "15 percent" → "15%" or vice versa
    - Ordinals: "first quarter" → "Q1" or "1st quarter"
    - Names: Keep names exact (do NOT rephrase names)

## Example
Chunk:
Company: AGILENT TECHNOLOGIES, INC.\nFiling Date: 2025-01-31\nForm: DEF 14A\nChunk 27\n---\n provides an overview of the compensation policies and practices applicable to our NEOs.\n\nTable of Contents:\n\nNamed Executive Officers\n\nExecutive Summary\n\nDetermining Executive pay\n\nFiscal year 2024 compensation\n\nOther compensation elements\n\nOurNEOsfor fiscal year 2024 are as follows:\n\nPadraig McDonnell, President and Chief Executive Officer (CEO)\n\nRobert McMahon, Senior Vice President, Chief Financial Officer (CFO)\n\nSimon May, Senior Vice President, President Diagnostic and Genomics Group (DGG)*\n\nHenrik Ancher-Jensen, Senior Vice President, President Order Fulfillment and Supply Chain (OFS)\n\nDominique Grau, Senior Vice President, Chief Human Resources Officer**\n\nMichael R. McMullen, Former President and Chief Executive Officer (CEO)***\n\n| *   \n **  | Mr. May joined Agilent on May 6, 2024.                                                                                                                                                                                                                                       \n Mr. Grau retired as an employee from Agilent on November 1, 2024.                                                                                                                                                                                                            |\n|:----|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n| *** | For fiscal year 2024, Mr. McDonnell was appointed CEO on May 1, 2024 while Mr. McMullen served as CEO from November 1, 2023 through April 30, 2024, and thereafter continued his employment as a Special Advisor to Mr. McDonnell through October 31, 2024, when he retired. |\n\nExecutive Summary\n\nLeadership Changes and Related Compensation\n\nFiscal year 2024 was a year of transition for Agilent during which the Board implemented an orderly CEO transition supported by its thoughtful and ongoing succession planning activities over several years. These transitions factored prominently in the Committee\u2019s compensation decisions as outlined in more detail below.\n\nChief Executive Officer Transition\n\nOn February 20, 2024, Mr. McMullen announced that he planned to retire as Agilent\u2019s CEO effective May 1, 2024. At that time, we named Mr. McDonnell, formerly our Chief Commercial Officer and President, Agilent CrossLab Group, as our COO, and CEO-Elect. Mr. McDonnell formally succeeded Mr. McMullen as Agilent\u2019s CEO on May 1, 2024.\n\nIn consultation with its independent compensation consultant, the Board recognized Mr. McDonnell\u2019s promotions to COO and CEO with two incremental pay actions which were designed to provide him with market-competitive annual compensation in both roles:\n\nUpon promotion to COO on February 20, 2024, Mr. McDon


Parts to focus on:
Dominique Grau, Senior Vice President, Chief Human Resources Officer
Mr. Grau retired as an employee from Agilent on November 1, 2024

Example output:
<task>
Which employee of Agilent Technologies, Inc. retired on 11/1/2024?
</task>
<answer>
Dominique Grau
</answer>

Note how the question:
- Includes the company name "Agilent Technologies, Inc." for context
- Rephrases "November 1, 2024" as "11/1/2024" to avoid keyword matching
- Keeps the person's name "Dominique Grau" exact in the answer

## Example 2
Chunk excerpt:
"...In Q3 2024, Apple Inc. reported revenue of $85.8 billion, representing a 6% increase year-over-year..."

Parts to focus on:
Q3 2024 revenue, $85.8 billion, 6% increase

Example output:
<task>
What was Apple Inc.'s revenue in the third quarter of 2024?
</task>
<answer>
$85.8 billion
</answer>

Note: "Q3 2024" is rephrased to "third quarter of 2024"

## Task
Here is the chunk:
{chunk_content}

Parts to focus on:
{parts_to_focus_on}

Output your response in the following format, no other text or formatting:
<task>{{The question that asks for the fact mentioned in the chunk content}}</task>
<answer>{{The exact answer to the question}}</answer>
"""
