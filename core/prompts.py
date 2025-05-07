# 🔍 検索クエリ再構成用プロンプト（再質問の明確化）
QUESTION_REPHRASE_PROMPT_STR = """会話履歴とユーザーの質問をもとに、検索用の問い合わせを生成してください。
会話履歴:
{{{chat_history}}}
質問:
{{{input}}}
適切な検索クエリ:
"""

# 🧠 日本語による簡潔な応答用プロンプト
DEFAULT_COMBINE_PROMPT_STR = """以下の文脈に基づいて、質問に日本語で簡潔に答えてください。
文脈:
{context}
質問:
{input}
"""

# 🧠 専門的で論理的な日本語の応答
JAPANESE_CONCISE_PROMPT_STR = """以下の情報に基づいて、**日本語で**専門的かつ論理的に、簡潔に答えてください。
文脈:
{context}
質問:
{input}
"""

# 🧠 英語での丁寧な応答
ENGLISH_VERBOSE_PROMPT_STR = """Based on the following context, please provide a detailed and logical answer in English.
Context:
{context}
Question:
{input}
"""
