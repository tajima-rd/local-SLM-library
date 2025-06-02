# -----------------------------
# 標準ライブラリ
# -----------------------------
import os
import json
import shutil
from uuid import uuid4
from pathlib import Path
from typing import Optional

# -----------------------------
# LangChain ライブラリ群
# -----------------------------
# Core
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # type: ignore

# Chains
from langchain.chains import (  # type: ignore
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.llm import LLMChain  # type: ignore
from langchain.chains.combine_documents.stuff import StuffDocumentsChain, create_stuff_documents_chain  # type: ignore
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain  # type: ignore
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain  # type: ignore

# Embeddings & VectorStore
from langchain_ollama import OllamaEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain.docstore.document import Document  # type: ignore

# -----------------------------
# 自作モジュール
# -----------------------------
from . import retriever_utils
from . import document_utils as docutils

from .retriever_utils import (
    edit_vectorstore_metadata,  # メタ編集関数
    RetrieverCategory,          # 階層カテゴリ定義
)


SUPPORTED_CHAINS = ["conversational", "retrievalqa", "llmchain", "map_reduce", "refine", "stuff"]

def get_chain(
    llm,
    chain_type: str,
    retriever,
    prompt_template: Optional[PromptTemplate] = None,
    chat_history_variable: str = "chat_history",
    k: int = 5,
    **kwargs
):
    """
    柔軟なチェーン構築関数（LangChain API対応）。
    
    Parameters:
    - llm: LLMインスタンス
    - chain_type: "conversational", "retrievalqa", "llmchain", "map_reduce", "refine", "stuff" のいずれか
    - retriever: 検索器
    - prompt_template: 任意のプロンプトテンプレート
    - chat_history_variable: 会話履歴の変数名
    - k: 検索数
    - kwargs: その他のオプション（未使用）
    """
    if chain_type == "conversational":
        return _build_conversational_chain(llm, retriever, prompt_template, chat_history_variable)

    elif chain_type == "retrievalqa":
        return _build_retrieval_qa_chain(llm, retriever, prompt_template)

    elif chain_type == "llmchain":
        if prompt_template is None:
            raise ValueError("prompt_template is required for llmchain")
        return prompt_template | llm  # RunnableSequence を返す

    elif chain_type in ["stuff", "map_reduce", "refine"]:
        return _build_summarize_chain(llm, chain_type, prompt_template)

    else:
        raise ValueError(f"Unsupported chain type: {chain_type}")

def _build_conversational_chain(
    llm,
    retriever,
    prompt_template: Optional[PromptTemplate],
    chat_history_variable: str = "chat_history"
):
    """
    会話履歴を考慮した Conversational Retrieval Chain を構築します。
    
    Parameters:
    - llm: 言語モデル
    - retriever: ベクトル検索器（Retriever）
    - prompt_template: combine_documents 用のプロンプト（任意）
    - chat_history_variable: 会話履歴変数名（default: "chat_history"）
    
    Returns:
    - LangChain Retrieval Chain
    """
    from prompts import QUESTION_REPHRASE_PROMPT_STR, DEFAULT_COMBINE_PROMPT_STR
    from langchain_core.prompts import ChatPromptTemplate # type: ignore

    # Step 1: 検索クエリを生成する History-aware Retriever の作成
    question_prompt = ChatPromptTemplate.from_template(QUESTION_REPHRASE_PROMPT_STR).partial(
        chat_history_variable=chat_history_variable
    )

    history_aware_retriever = create_history_aware_retriever(
        retriever=retriever,
        llm=llm,
        prompt=question_prompt
    )

    # Step 2: combine_documents チェーンの構築
    if prompt_template is None:
        prompt_template = ChatPromptTemplate.from_template(DEFAULT_COMBINE_PROMPT_STR)
    
    combine_chain = create_stuff_documents_chain(llm, prompt_template)

    # Step 3: 全体チェーンを構築して返す
    return create_retrieval_chain(history_aware_retriever, combine_chain)

def _build_retrieval_qa_chain(
    llm,
    retriever,
    prompt_template: Optional[PromptTemplate]
):
    """
    シンプルな Retrieval QA チェーンを構築します。
    
    Parameters:
    - llm: 言語モデル
    - retriever: 検索器
    - prompt_template: combine_documents 用のプロンプト（None の場合はデフォルトを使用）

    Returns:
    - LangChain Retrieval Chain
    """
    if prompt_template is None:
        prompt_template = DEFAULT_COMBINE_PROMPT

    combine_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, combine_chain)

def _build_summarize_chain(
    llm,
    chain_type: str,
    prompt_template: Optional[PromptTemplate]
):
    """
    要約チェーンを構築します。stuff / map_reduce / refine の3種に対応。

    Parameters:
    - llm: 言語モデル
    - chain_type: "stuff", "map_reduce", "refine"
    - prompt_template: combine_documents 用のプロンプト（任意）

    Returns:
    - 文書要約チェーン
    """
    if prompt_template is None:
        prompt_template = ChatPromptTemplate.from_template("""
        以下の複数文書を日本語で要約してください：

        {context}
        """)

    if chain_type == "stuff":
        return create_stuff_documents_chain(llm, prompt_template)
    elif chain_type == "map_reduce":
        return _build_map_reduce_chain(llm, prompt_template)
    elif chain_type == "refine":
        return create_refine_documents_chain(llm, prompt_template)
    else:
        raise ValueError(f"Unknown summarize chain type: {chain_type}")

def _build_map_reduce_chain(
    llm,
    prompt_template: PromptTemplate
):
    """
    MapReduce 形式の要約チェーンを構築します。

    Parameters:
    - llm: 言語モデル
    - prompt_template: PromptTemplate インスタンス

    Returns:
    - MapReduceDocumentsChain
    """
    # LLM チェーン（Mapping フェーズ）
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Stuff チェーン（文書統合用）
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=prompt_template,
        document_variable_name="context"
    )

    # Reduce チェーン（要約の集約）
    reduce_chain = ReduceDocumentsChain(
        combine_documents_chain=stuff_chain
    )

    # MapReduce チェーンの構築
    return MapReduceDocumentsChain(
        llm_chain=llm_chain,
        reduce_documents_chain=reduce_chain
    )

def prepare_chain_from_path(
    llm,
    faiss_paths: list[Path],
    chain_type: str = "conversational",
    k: int = 5,
    prompt_template: Optional[PromptTemplate] = None,
):
    """
    指定パス配下のすべての FAISS ベクトルストアを統合し、Retriever および RAG チェーンを構築する。

    Parameters:
    - llm: 言語モデル
    - base_path: ベクトルストアのディレクトリ（.faiss を含むフォルダ群の親）
    - chain_type: チェーンの種類（"conversational", "retrievalqa", など）
    - k: 検索数（Retriever用）
    - prompt_template: combine_documents 用のプロンプト（任意）

    Returns:
    - LangChain チェーンオブジェクト
    """
    if not faiss_paths:
        raise FileNotFoundError(f"ベクトルストア（.faiss）が見つかりません")

    print(f"🔍 {len(faiss_paths)} 件のベクトルストアを統合します: {[str(p) for p in faiss_paths]}")

    vectorstore = retriever_utils.merge_vectorstore([str(p) for p in faiss_paths])
    retriever = retriever_utils.create_retriever(vectorstore, k=k)

    return get_chain(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        prompt_template= prompt_template,
        k=k,
    )


def save_chain_from_markdown(
    md_path: str,
    vect_path: str,
    embedding_name: str,
    category: RetrieverCategory,
    loader_type: str = "markdown",
):
    
    # 埋め込みモデルの初期化
    embeddings = OllamaEmbeddings(model=embedding_name)

    documents = docutils.load_documents(md_path, loader_type)

    print("テキストを分割してチャンクを割り当てます。")
    splitter = docutils.suggest_text_splitter(
        documents=documents,
        loader_type=loader_type
    )
    split_docs = splitter.split_documents(documents)

    for doc in split_docs:
        doc.metadata["doc_id"] = str(uuid4())

        existing_category = doc.metadata.get("category", {})
        if not isinstance(existing_category, dict):
            existing_category = {}

        doc.metadata["category"] = category.to_dict()

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(vect_path)

    metadata = {
        "embedding_model": embeddings.model,
        "loader_type": loader_type,
        "category": category.to_dict(),
    }
    with open(os.path.join(vect_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return True

def save_chain_from_text(
    text: str,
    vect_path: str,
    embedding_name: str,
    category: RetrieverCategory,
    loader_type: str = "text",  # 明示しておくとよい
) -> bool:
    """
    プレーンテキスト文字列からドキュメントを生成し、FAISS に保存する。
    """

    # 埋め込みモデルの初期化
    embeddings = OllamaEmbeddings(model=embedding_name)

    # Documentに変換（metadataはあとで付与）
    document = Document(page_content=text, metadata={})
    documents = [document]

    print("テキストを分割してチャンクを割り当てます。")
    splitter = docutils.suggest_text_splitter(
        documents=documents,
        loader_type=loader_type
    )
    split_docs = splitter.split_documents(documents)

    for doc in split_docs:
        doc.metadata["doc_id"] = str(uuid4())

        existing_category = doc.metadata.get("category", {})
        if not isinstance(existing_category, dict):
            existing_category = {}

        doc.metadata["category"] = category.to_dict()

    # FAISS保存
    print(f"[DEBUG] チャンク数: {len(split_docs)}")
    print(f"[DEBUG] 最初のチャンク: {split_docs[0].page_content if split_docs else 'なし'}")

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(vect_path)

    # メタデータの保存
    metadata = {
        "embedding_model": embeddings.model,
        "loader_type": loader_type,
        "category": category.to_dict(),
    }
    with open(os.path.join(vect_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return True