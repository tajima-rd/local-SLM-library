import json
import rag
from prompts import QUESTION_REPHRASE_PROMPT, DEFAULT_COMBINE_PROMPT # type: ignore

from pathlib import Path
from typing import Optional

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate # type: ignore
from langchain_core.runnables import RunnableSequence # type: ignore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain # type: ignore
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain # type: ignore


SUPPORTED_CHAINS = ["conversational", "retrievalqa", "llmchain", "map_reduce", "refine", "stuff"]

from langchain.memory import ConversationBufferMemory # type: ignore

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
    最新 LangChain API に対応した柔軟なチェーン構築関数。
    - conversational: history-aware retriever + QA chain
    - retrievalqa: retriever + QA chain
    - llmchain: 単独 LLMChain
    - map_reduce/refine/stuff: 文書要約
    """
    if chain_type == "conversational":
        return _build_conversational_chain(llm, retriever, prompt_template, chat_history_variable)

    elif chain_type == "retrievalqa":
        return _build_retrieval_qa_chain(llm, retriever, prompt_template)

    elif chain_type == "llmchain":
        if prompt_template is None:
            raise ValueError("prompt_template is required for llmchain")
        return prompt_template | llm  # RunnableSequence

    elif chain_type in ["stuff", "map_reduce", "refine"]:
        return _build_summarize_chain(llm, chain_type, prompt_template)

    else:
        raise ValueError(f"Unsupported chain type: {chain_type}")

# Step 2: Conversational Retrieval Chain の構築
def _build_conversational_chain(llm, retriever, prompt_template, chat_history_variable):
    question_prompt = QUESTION_REPHRASE_PROMPT.partial(chat_history_variable=chat_history_variable)

    history_aware_retriever = create_history_aware_retriever(
        retriever=retriever,
        llm=llm,
        prompt=question_prompt  # ✅ prompt を使う
    )

    if prompt_template is None:
        combine_chain = create_stuff_documents_chain(llm, DEFAULT_COMBINE_PROMPT)
    else:
        combine_chain = create_stuff_documents_chain(llm, prompt_template)

    return create_retrieval_chain(history_aware_retriever, combine_chain)

# Step 3: RetrievalQA チェーンの構築
def _build_retrieval_qa_chain(llm, retriever, prompt_template):
    # prompt_template が None の場合は既定のものを使う
    if prompt_template is None:
        prompt_template = DEFAULT_COMBINE_PROMPT

    combine_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, combine_chain)

def _build_summarize_chain(llm, chain_type: str, prompt_template: Optional[PromptTemplate]):
    if prompt_template is None:
        prompt_template = ChatPromptTemplate.from_template("""
        以下の複数文書を日本語で要約してください：

        {context}
        """)

    if chain_type == "stuff":
        return create_stuff_documents_chain(llm, prompt_template)
    elif chain_type == "map_reduce":
        return create_map_reduce_documents_chain(llm, prompt_template)
    elif chain_type == "refine":
        return create_refine_documents_chain(llm, prompt_template)
    else:
        raise ValueError(f"Unknown summarize chain type: {chain_type}")

def _build_map_reduce_chain(llm, prompt_template):
    from langchain.chains.llm import LLMChain
    from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
    from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain_core.prompts import PromptTemplate

    # プロンプト処理
    prompt = PromptTemplate.from_template(prompt_template) if isinstance(prompt_template, str) else prompt_template

    # LLM チェーン
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Stuff チェーン（文書の統合）
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=prompt,
        document_variable_name="context"
    )

    # Reduce チェーン
    reduce_chain = ReduceDocumentsChain(
        combine_documents_chain=stuff_chain
    )

    # MapReduce チェーン
    return MapReduceDocumentsChain(
        llm_chain=llm_chain,
        reduce_documents_chain=reduce_chain
    )

def prepare_chain_for_category(
    llm,
    category: str,
    base_path: Path,
    memory=None,
    chain_type: str = "conversational",
    k: int = 5,
    prompt_template: Optional[PromptTemplate] = None,
):
    """カテゴリに対応するvectorstore群をマージし、retrieverとchainを構築"""
    all_faiss_paths = list(base_path.glob("**/*.faiss"))
    vect_paths = []

    for path in all_faiss_paths:
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("category") == category:
            vect_paths.append(str(path))

    if not vect_paths:
        raise FileNotFoundError(f"カテゴリ '{category}' に一致するベクトルストアが見つかりません: {base_path}")

    print(f"🧩 カテゴリ '{category}' に一致するベクトルストア {len(vect_paths)} 件: {vect_paths}")
    vectorstore = rag.merge_vectorstore(vect_paths)
    retriever = rag.create_retriever(vectorstore, k=k)

    return get_chain(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        memory=memory,
        prompt_template=prompt_template,
        k=k,
    )
