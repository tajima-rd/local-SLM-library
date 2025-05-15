# chain_factory.py

import json
import retriever_utils

from pathlib import Path
from typing import Optional

# LangChain core modules
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate # type: ignore

# LangChain chains
from langchain.chains import ( # type: ignore
    create_history_aware_retriever,
    create_retrieval_chain
) 
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain # type: ignore
from langchain.chains.llm import LLMChain # type: ignore
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain # type: ignore
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain # type: ignore
from langchain.chains.combine_documents.stuff import StuffDocumentsChain # type: ignore

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

def prepare_chain_for_category(
    llm,
    category: retriever_utils.RetrieverCategory,
    base_path: Path,
    chain_type: str = "conversational",
    k: int = 5,
    prompt_template: Optional[PromptTemplate] = None,
):
    """
    指定カテゴリに対応するベクトルストアを統合し、Retriever およびチェーンを構築する。

    Parameters:
    - llm: 言語モデル
    - category: 対象カテゴリ名（metadata.json の "category" に対応）
    - base_path: ベクトルストア群のベースディレクトリ
    - chain_type: チェーンの種類（例: conversational）
    - k: 検索数（Retriever用）
    - prompt_template: プロンプトテンプレート（任意）

    Returns:
    - LangChain チェーンオブジェクト
    """
    all_faiss_paths = list(base_path.glob("**/*.faiss"))
    vect_paths = []

    for path in all_faiss_paths:
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("category", {}).get("tagname") == category.tagname:
            vect_paths.append(str(path))

    if not vect_paths:
        raise FileNotFoundError(f"カテゴリ '{category}' に一致するベクトルストアが見つかりません: {base_path}")

    print(f"🧩 カテゴリ '{category}' に一致するベクトルストア {len(vect_paths)} 件: {vect_paths}")
    
    # Retriever を構築
    vectorstore = retriever_utils.merge_vectorstore(vect_paths)
    retriever = retriever_utils.create_retriever(vectorstore, k=k)

    return get_chain(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        prompt_template=prompt_template,
        k=k,
    )
