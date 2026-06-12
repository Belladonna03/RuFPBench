EPSTEIN_TRUTH_TYPES = [
    "person",
    "location",
    "event",
    "organization",
    "date/time"
]

EPSTEIN_EXPLORATION_PROMPT = """
You need to create a challenging search question based on emails.

Given seed email threads and a truth type, explore the email inbox to discover related information. Create a multi-hop search question with 3 clues pointing to a single truth. At least one of the clues should contain the truth, and there should be 3 unique email threads in total.

**Goal**: 3 clues → 3 unique email threads → 1 truth (matching the truth type)

Aim to finish in 10-15 tool calls.

---

## Rules for Clues and Question

### 1. NO NAMES - EVER
Never include names of people, places, or organizations in clues or questions. Always describe instead:
- BAD: "Sydney" → GOOD: "an Australian city"
- BAD: "Goldman Sachs" → GOOD: "a major investment bank"
- BAD: "Jeffrey Epstein" → GOOD: "a person mentioned to be convicted of this crime"
-

### 2. NO SEARCHABLE KEYWORDS
Avoid any word that would directly match the target email. Ensure that no keyword matches exist:
- No proper nouns, specific dates, or unique phrases from the email
- No domain-specific terminology that's rare in the corpus
- BAD: "November 2016" → GOOD: "Late 2016"
- BAD: "2023" → GOOD: "Early 2020s"

### 3. NO COMPUTATIONS
Never require the reader to perform any mathematical operations to understand a clue or arrive at the truth. State facts directly without encoding them as puzzles.
- BAD: "reached the age where one subtracts one from a perfect square of eight" → GOOD: "in their early sixties"
- BAD: "exactly half a dozen years after the millennium" → GOOD: "in the mid-2000s"
- BAD: "the sum of 30 and 3 people attended" → GOOD: "over thirty people attended"

### 4. DO NOT ASSUME KNOWLEDGE ABOUT SPECIFIC PEOPLE
The searcher will not know facts about individuals mentioned in the emails (their occupation, relationships, history, etc.). Any information about a person needed to solve the puzzle must be discoverable within the email corpus itself.
- BAD: Describing someone as "a lawyer" when their profession is only known outside the emails -> GOOD: Describing someone by actions visible in the emails, like "an individual who handled this trial"

### 5. MULTIPLE SEARCHES NECESSARY
Each clue should need 2-3 search attempts. Connect clues through relationships:
- "mentioned in the same thread as..."
- "asked about by the same person who..."

---

## Available Tools
- `hybrid_search_across_all(query)` - Semantic + keyword search. Use natural language.
- `grep_across_all(pattern)` - Regex search for exact patterns/names.
- `search_across_person(person, query)` - Search within a specific person's emails.
- `get_random_across_person(person)` - Random samples from a person's emails.
- `get_thread(thread_id)` - Read full thread content.

---

Example output (from truth type: person):
<clues>
There is an individual who handled a trial in an Austrailian city in the late 2010s. At a later time, this person asked about a girl who was mentioned to have flown to 5 cities across 3 countries. One of these cities (city A) is a European city, which is mentioned in another email thread concerning a writing piece. This writing piece mentions city A as being less than 100 miles away from a prison controlled by a conservative government.
<question>
Who is the author of this piece?
</question>
<truth>
David Leonhardt
</truth>
<supporting_items>
    <item>
        <id>
        472
        </id>
        <reasoning>
        Weingarten, Reid was handling a trial in Sydney (Austrailian city) in 2017 (late 2010s).
        </reasoning>
    </item>
    <item>
        <id>
        80
        </id>
        <reasoning>
        Weingarten, Reid asked about a girl who flew to London, DC, Dallas, Seatle, and Tel Aviv.
        </reasoning>
    </item>
    <item>
        <id>
        1466
        </id>
        <reasoning>
        A macroeconomics piece mentions David Cameron's Conservative government in Britain testing an idea at a prison 75 miles north of London (the European city).
        </reasoning>
    </item>
</supporting_items>
------------------------------------------------------------------------------------------------

Here is your task:
Seed threads:
{seed_threads}

Truth type (the truth should fall under this category):
{truth_type}

------------------------------------------------------------------------------------------------
Output your response in the following format, no other text or formatting:
<clues>
{{The 3 subtle clues which point to the common truth. Keep them concise, 3 sentences max.}}
</clues>
<question>
{{The question which asks for the common truth.}}
</question>
<truth>
{{The one and only exact truth to the question.}}
</truth>
<supporting_items>
    <item>
        <id>
        {{ID of the thread for clue 1}}
        </id>
        <reasoning>
        {{Reasoning for why this thread supports the clue and helps lead to the truth}}
        </reasoning>
    </item>
    <item>
        <id>
        {{ID of the thread for clue 2}}
        </id>
        <reasoning>
        {{Reasoning for why this thread supports the clue and helps lead to the truth}}
        </reasoning>
    </item>
    <item>
        <id>
        {{ID of the thread for clue 3}}
        </id>
        <reasoning>
        {{Reasoning for why this thread supports the clue and helps lead to the truth}}
        </reasoning>
    </item>
</supporting_items>
"""

EPSTEIN_EXTRACTION_PROMPT_SINGLE = """
You are extracting supporting evidence for a search task based on a single email thread.

Given the clues, question, truth (answer), and the content of one supporting email thread:
- Extract the relevant quotes from this thread
- Extract the targeted clue that the quote supports
- Explain why the quote supports the targeted clue

## Guidelines
- Each quote should be a direct excerpt from the source content (should be matched via substring search)
- Keep each quote concise (typically a few words or a sentence)
- If the relevant information comes from multiple parts of the thread, provide multiple separate quotes
- If the relevant quotes match multiple parts of the clue, provide multiple separate quotes
- The reasoning should explain how the quote supports the targeted clue
- IMPORTANT: If the thread does NOT actually contain content relevant to the clues, output None for clue_quotes and item_quotes (e.g., `<clue_quotes>None</clue_quotes>`). This indicates the thread is not a valid supporting item.

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
<thread>
    <id>{thread_id}</id>
    <reasoning>{reasoning}</reasoning>
    <content>
{content}
    </content>
</thread>

## Output Format
Output exactly one item for this thread:
<item>
    <id>
    {thread_id}
    </id>
    <clue_quotes>
        {{Either <q>exact substring from clues</q> OR None if thread is not relevant}}
    </clue_quotes>
    <item_quotes>
        {{Either <q>exact quote(s) from thread</q> OR None if thread is not relevant}}
    </item_quotes>
    <reasoning>
    {{Reasoning for why this thread supports the clue, or why it is not relevant}}
    </reasoning>
    <contains_truth>{{true or false}}</contains_truth>
    <truth_quotes>
        <q>{{The exact truth as it appears in this thread, or None if contains_truth is false}}</q>
    </truth_quotes>
</item>
"""

EPSTEIN_COHERENCE_CHECK_PROMPT = """
You are verifying that the supporting evidence for a search task is internally coherent and logically connected.

Given the clues, question, truth (answer), and the supporting items (each with their content and reasoning), determine whether all the supporting items connect together and make sense as a coherent chain of evidence.

## What to Check
1. **Entity Consistency**: When the clues reference the same entity (e.g., "a legal professional", "the same reporter"), verify that the supporting documents are actually referring to the same entity, not different ones.
2. **Logical Flow**: The supporting items should form a logical chain that leads from the clues to the truth.
3. **No Contradictions**: The supporting items should not contradict each other or the clues.
4. **Complete Coverage**: Together, the supporting items should cover all the key claims in the clues.

## Example of Incoherence
If the clues say: "That reporter at the political news organization was following litigation and contacted a legal professional about a court filing. The same legal professional who was contacted later..."
And two supporting docs each mention a different legal professional, this would be INCOHERENT because the clues imply it's the same legal professional throughout.

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
<supporting_items>
{supporting_items_formatted}
</supporting_items>

## Output Format
<reasoning>
{{Detailed analysis of whether the supporting items are coherent and logically connected. Identify any inconsistencies, contradictions, or gaps. Be specific about which entities/facts you're checking for consistency.}}
</reasoning>
<coherent>{{true or false}}</coherent>
"""
