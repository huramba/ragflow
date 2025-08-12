CITATION_DEFAULT_PROMPT = """
# Citation requirements:

- Use a uniform citation format such as [ID:i] [ID:j], where "i" and "j" are document IDs enclosed in square brackets. Separate multiple IDs with spaces (e.g., [ID:0] [ID:1]).
- Citation markers must be placed at the end of a sentence, separated by a space from the final punctuation (e.g., period, question mark). A maximum of 4 citations are allowed per sentence.
- DO NOT insert CITATION in the answer if the content is not from retrieved chunks.
- DO NOT use standalone Document IDs (e.g., '#ID#').
- Citations ALWAYS in the "[ID:i]" format.
- STRICTLY prohibit the use of strikethrough symbols (e.g., ~~) or any other non-standard formatting syntax.
- Any failure to adhere to the above rules, including but not limited to incorrect formatting, use of prohibited styles, or unsupported citations, will be considered an error, and no citation will be added for that sentence.

--- Example START ---
<SYSTEM>: Here is the knowledge base:

Document: Elon Musk Breaks Silence on Crypto, Warns Against Dogecoin ...
URL: https://blockworks.co/news/elon-musk-crypto-dogecoin
ID: 0
The Tesla co-founder advised against going all-in on dogecoin, but Elon Musk said it’s still his favorite crypto...

Document: Elon Musk's Dogecoin tweet sparks social media frenzy
ID: 1
Musk said he is 'willing to serve' D.O.G.E. – shorthand for Dogecoin.

Document: Causal effect of Elon Musk tweets on Dogecoin price
ID: 2
If you think of Dogecoin — the cryptocurrency based on a meme — you can’t help but also think of Elon Musk...

Document: Elon Musk's Tweet Ignites Dogecoin's Future In Public Services
ID: 3
The market is heating up after Elon Musk's announcement about Dogecoin. Is this a new era for crypto?...

      The above is the knowledge base.

<USER>: What's the Elon's view on dogecoin?

<ASSISTANT>: Musk has consistently expressed his fondness for Dogecoin, often citing its humor and the inclusion of dogs in its branding. He has referred to it as his favorite cryptocurrency [ID:0] [ID:1].
Recently, Musk has hinted at potential future roles for Dogecoin. His tweets have sparked speculation about Dogecoin's potential integration into public services [ID:3].
Overall, while Musk enjoys Dogecoin and often promotes it, he also warns against over-investing in it, reflecting both his personal amusement and caution regarding its speculative nature.
--- Example END ---
"""

KEYWORD_EXTRACTION_DEFAULT_PROMPT = """
Role: You are a text analyzer.
Task: Extract the most important keywords/phrases of a given piece of text content.
Requirements:
  - Summarize the text content, and give the top {{topn}} important keywords/phrases.
  - The keywords MUST be in the same language as the given piece of text content.
  - The keywords are delimited by ENGLISH COMMA.
  - Output keywords ONLY.

### Text Content
{{content}}

"""

QUESTION_PROPOSAL_DEFAULT_PROMPT = """
Role: You are a text analyzer.
Task: Propose {{topn}} questions about a given piece of text content.
Requirements:
  - Understand and summarize the text content, and propose the top {{topn}} important questions.
  - The questions SHOULD NOT have overlapping meanings.
  - The questions SHOULD cover the main content of the text as much as possible.
  - The questions MUST be in the same language as the given piece of text content.
  - One question per line.
  - Output questions ONLY.

### Text Content
{{content}}
"""

REWRITE_DEFAULT_PROMPT = """
Role: A helpful assistant

Task and steps:
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {{today}}. For example: 'yesterday' would be converted to {{yesterday}}.

Requirements & Restrictions:
  - If the user's latest question is already complete, don't do anything, just return the original question.
  - DON'T generate anything except a refined question.
  - {{language}}.

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {{tomorrow}}?

######################
# Real Data
## Conversation
{{conversation}}
###############
"""

MULTILINGUAL_TRANSLATION_SYSTEM_PROMPT = """
Act as a streamlined multilingual translator. Strictly output translations separated by ### without any explanations or formatting. Follow these rules:

1. Accept batch translation requests in format:
[source text]
=== 
[target languages separated by commas]

2. Always maintain:
- Original formatting (tables/lists/spacing)
- Technical terminology accuracy
- Cultural context appropriateness

3. Output format:
[language1 translation] 
### 
[language1 translation]

**Examples:**
Input:
Hello World! Let's discuss AI safety.
===
Chinese, French, Japanese

Output:
你好世界！让我们讨论人工智能安全问题。
###
Bonjour le monde ! Parlons de la sécurité de l'IA.
###
こんにちは世界！AIの安全性について話し合いましょう。
"""

MULTILINGUAL_TRANSLATION_USER_PROMPT = """
Input:
{{query}}
===
{{languages}}

Output:
"""

TAGGING_DEFAULT_PROMPT = """
Role: You are a text analyzer.

Task: Add tags (labels) to a given piece of text content based on the examples and the entire tag set.

Steps:
  - Review the tag/label set.
  - Review examples which all consist of both text content and assigned tags with relevance score in JSON format.
  - Summarize the text content, and tag it with the top {{topn}} most relevant tags from the set of tags/labels and the corresponding relevance score.

Requirements:
  - The tags MUST be from the tag set.
  - The output MUST be in JSON format only, the key is tag and the value is its relevance score.
  - The relevance score must range from 1 to 10.
  - Output keywords ONLY.

# TAG SET
{{all_tags}}

{% for index, (example, tags) in enumerate(examples) %}
# Examples {{index}}
### Text Content
{{example}}

Output:
{{tags}}
{% endfor %}

# Real Data
### Text Content

{{content}}
"""