import os
import construct # type: ignore
import chain
import rag
from pathlib import Path
from langchain_ollama import OllamaLLM # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore


# --- ディレクトリ設定 ---
base_dir = Path("/home/yufujimoto/Documents/Projects/生成系AI/LocalLLM/webui/modules/core/sample")
markdown_dir = base_dir / "markdown"
vectorstore_dir = base_dir / "vectorstore"

# --- 入力ファイル ---
input_files = [
    base_dir / "cat_tools.md",
    base_dir / "japan_catapillar.md",
    base_dir / "Proffesional_College_of_arts_and_tourism.md",
]

# --- カテゴリ指定 ---
category = "test"

# --- ベクトルストアの構築 ---
for in_file in input_files:
    basename = in_file.stem
    md_path = markdown_dir / f"{basename}.md"
    vect_path = vectorstore_dir / f"{basename}.faiss"
    construct.vectorization(in_file, md_path, vect_path, category=category)

# --- モデル定義 ---
llm = OllamaLLM(model="granite3.2:8b")

CUSTOM_PROMPT = ChatPromptTemplate.from_template("""
        以下の情報に基づいて、**日本語で**専門的かつ論理的に、簡潔に答えてください。

        文脈:
        {context}

        質問:
        {input}
    """)

# --- チェーンの構築（カテゴリ指定） ---
qa_chain = chain.prepare_chain_for_category(
    llm=llm,
    category=category,
    base_path=vectorstore_dir,
    chain_type="conversational",
    prompt_template=CUSTOM_PROMPT,
    k=5,
)

# --- 対話ループ ---
print("RAG 実験モード開始（終了するには 'exit' と入力）")
while True:
    query = input("\n🗨 質問してください: ")
    if query.strip().lower() == "exit":
        break

    response = qa_chain.invoke({
        "input": query,
        "chat_history": []
    })

    for doc in response.get("source_documents", []):
        print("📄", doc.metadata.get("source"))

    print("\n🧠 回答:")
    print(response.get("answer") or response.get("output"))
