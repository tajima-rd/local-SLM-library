# prompts.py

from langchain_core.prompts import ChatPromptTemplate

# 🔍 会話履歴を用いた検索クエリ再構成プロンプト
QUESTION_REPHRASE_PROMPT = ChatPromptTemplate.from_template("""
        会話履歴とユーザーの質問をもとに、検索用の問い合わせを生成してください。

        会話履歴:
        {{{chat_history}}}

        質問:
        {{{input}}}

        適切な検索クエリ:
    """)

# 🧠 文脈に基づいて質問に回答するための基本プロンプト
DEFAULT_COMBINE_PROMPT = ChatPromptTemplate.from_template("""
        以下の文脈に基づいて、質問に日本語で簡潔に答えてください。

        文脈:
        {context}

        質問:
        {input}
    """)
