# -----------------------------
# 標準ライブラリ
# -----------------------------
from uuid import uuid4
from pathlib import Path
from typing import Any, Optional, List, Dict

# -----------------------------
# サードパーティライブラリ (LangChain)
# -----------------------------
# Core
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate # type: ignore
from langchain_core.runnables import Runnable # チェーンは Runnable の一種 # type: ignore
from langchain_core.output_parsers import StrOutputParser # LLM 出力を文字列にパース # type: ignore

# Chains (Legacy or specific utility functions)
# RetrievalQA はレガシーですが、まだ利用されるため残します（推奨は create_*_chain 系）
from langchain.chains import ( # type: ignore
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.llm import LLMChain # type: ignore
from langchain.chains.combine_documents.stuff import StuffDocumentsChain, create_stuff_documents_chain # type: ignore
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain # type: ignore
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain # type: ignore

# Embeddings & VectorStore
from langchain_community.embeddings import OllamaEmbeddings # type: ignore # OllamaEmbeddings は community に移動
from langchain_community.vectorstores import FAISS # type: ignore # FAISS は community に移動
from langchain_core.documents import Document # type: ignore # Document は core に移動

# -----------------------------
# 自作モジュール
# -----------------------------
from . import document_utils as docutils

# prompts.py から定数をインポート (外部ファイルに依存)
try:
    from .prompts import QUESTION_REPHRASE_PROMPT_STR, DEFAULT_COMBINE_PROMPT_STR
except ImportError:
    # prompts.py が存在しない場合の代替定義（またはエラー）
    print("⚠️ prompts.py が見つかりません。デフォルトのプロンプト文字列を使用します。")
    QUESTION_REPHRASE_PROMPT_STR = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""

    DEFAULT_COMBINE_PROMPT_STR = """Answer the following question based only on the provided context:

{context}

Question: {input}
"""

# -----------------------------
# 定数
# -----------------------------
# サポートするチェーンタイプ
SUPPORTED_CHAINS = ["conversational", "retrievalqa", "llmchain", "stuff", "map_reduce", "refine"]

# -----------------------------
# チェーン構築関数
# -----------------------------

def get_chain(
    llm: Any, # 汎用的なLLMオブジェクト (invokeメソッドを持つことを想定)
    chain_type: str,
    retriever: Optional[Any] = None, # retriever_chain 以外では None の場合がある
    prompt_template: Optional[BasePromptTemplate] = None, # より具体的な型に
    chat_history_variable: str = "chat_history",
    k: int = 5, # retriever の k 値を渡せるように
    **kwargs: Any # その他のチェーン固有のオプション
) -> Runnable: # Runnable インターフェースを返すことを明示
    """
    柔軟なチェーン構築関数（LangChain Runnable API対応）。
    
    Parameters:
    - llm: LLMインスタンス (invokeメソッドを持つオブジェクト)
    - chain_type: "conversational", "retrievalqa", "llmchain", "stuff", "map_reduce", "refine" のいずれか
    - retriever: 検索器 (Retriever) オブジェクト。Retrieval系のチェーンで必須。
    - prompt_template: 任意のプロンプトテンプレート (BasePromptTemplate)。チェーンタイプによっては必須または推奨。
    - chat_history_variable: 会話履歴の変数名 (conversationalチェーン用)
    - k: Retrieverの検索数 (主に prepare_chain_from_path で使用されるが、引数として保持)
    - kwargs: その他のオプション (チェーン構築時の追加引数など)

    Returns:
    - 構築された LangChain チェーンオブジェクト (Runnable)

    Raises:
        ValueError: サポートされていないチェーンタイプが指定された場合、または必須引数が不足している場合。
    """
    if chain_type not in SUPPORTED_CHAINS:
        raise ValueError(f"Unsupported chain type: {chain_type}. Supported types are: {SUPPORTED_CHAINS}")

    # Retriever が必要なチェーンタイプをチェック
    retrieval_required = chain_type in ["conversational", "retrievalqa"]
    if retrieval_required and retriever is None:
         raise ValueError(f"retriever is required for chain type: {chain_type}")

    if chain_type == "conversational":
        # conversational chain は Runnable API で構築
        return _build_conversational_chain(
            llm=llm,
            retriever=retriever, # retriever は必須
            prompt_template=prompt_template,
            chat_history_variable=chat_history_variable
        )

    elif chain_type == "retrievalqa":
        # retrievalqa chain も Runnable API で構築 (旧 RetrievalQA クラスとは異なる)
        return _build_retrieval_qa_chain(
            llm=llm,
            retriever=retriever, # retriever は必須
            prompt_template=prompt_template
        )

    elif chain_type == "llmchain":
        # llmchain は単なる LLM と Optional のプロンプトの組み合わせ (RunnableSequence)
        if prompt_template is None:
            # LLM単体でもRunnableだが、通常LLMChainはプロンプトとセット
            print("⚠️ llmchain タイプですが prompt_template が指定されていません。LLM単体が返されます。")
            return llm
        return prompt_template | llm | StrOutputParser() # LLMChain の代わり # type: ignore

    elif chain_type in ["stuff", "map_reduce", "refine"]:
        # 要約・文書結合チェーン
        return _build_combine_documents_chain(
            llm=llm,
            chain_type=chain_type,
            prompt_template=prompt_template
        )

    # Note: prepare_chain_from_retriever が削除されたため、
    # ここで RetrievalQA.from_chain_type を直接使うロジックは不要になりました。
    # 必要に応じて上記チェーンタイプで対応します。

def _build_conversational_chain(
    llm: Any,
    retriever: Any,
    prompt_template: Optional[BasePromptTemplate],
    chat_history_variable: str = "chat_history"
) -> Runnable:
    """
    会話履歴を考慮した Conversational Retrieval Chain を Runnable API で構築します。
    
    Parameters:
    - llm: 言語モデル
    - retriever: ベクトル検索器（Retriever）
    - prompt_template: combine_documents 用のプロンプト（任意）。ChatPromptTemplate推奨。
    - chat_history_variable: 会話履歴変数名（default: "chat_history"）
    
    Returns:
    - LangChain Retrieval Chain (Runnable)
    """
    # Step 1: 検索クエリを生成する History-aware Retriever の作成
    # Runnable API では、入力 (例: {"input": ..., "chat_history": ...}) を受け取り、
    # 変換されたクエリ文字列を返すチェーンを作成します。
    question_generator_prompt = ChatPromptTemplate.from_template(QUESTION_REPHRASE_PROMPT_STR).partial(
        chat_history_variable=chat_history_variable
    )

    # create_history_aware_retriever は、変換されたクエリを使ってRetrieverを実行する Runnable を返します。
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=question_generator_prompt
    )

    # Step 2: combine_documents チェーンの構築
    # 検索結果 (documents) と元の入力 (input) を受け取り、最終的な回答を生成するチェーン
    if prompt_template is None:
        # デフォルトの結合プロンプト
        prompt_template = ChatPromptTemplate.from_template(DEFAULT_COMBINE_PROMPT_STR)
    elif not isinstance(prompt_template, BasePromptTemplate):
         print(f"⚠️ 指定された prompt_template の型が無効です ({type(prompt_template)})。ChatPromptTemplate を想定しています。")
         prompt_template = ChatPromptTemplate.from_template(DEFAULT_COMBINE_PROMPT_STR) # デフォルトに戻すか警告

    # create_stuff_documents_chain は、文書リストと入力をStuffしてプロンプトに渡し、LLMを呼び出す Runnable を返します。
    combine_chain = create_stuff_documents_chain(llm, prompt_template)

    # Step 3: History-aware Retriever と Combine Chain を結合して、Retrieval Chain を構築
    # create_retrieval_chain は、Retriever Runnable と CombineDocumentsChain Runnable を結合し、
    # 入力 {"input": ..., "chat_history": ...} を受け取って、
    # 中間結果 (retrieved_documents) と最終結果 (answer) を含む辞書を返す Runnable を構築します。
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_chain)

    return retrieval_chain

def _build_retrieval_qa_chain(
    llm: Any,
    retriever: Any,
    prompt_template: Optional[BasePromptTemplate]
) -> Runnable:
    """
    シンプルな Retrieval QA チェーンを Runnable API で構築します。会話履歴は考慮しません。
    
    Parameters:
    - llm: 言語モデル
    - retriever: 検索器
    - prompt_template: combine_documents 用のプロンプト（任意）。ChatPromptTemplate推奨。

    Returns:
    - LangChain Retrieval Chain (Runnable)
    """
    # Step 1: combine_documents チェーンの構築
    # 検索結果 (documents) と元の入力 (input) を受け取り、最終的な回答を生成するチェーン
    if prompt_template is None:
        # デフォルトの結合プロンプト
        prompt_template = ChatPromptTemplate.from_template(DEFAULT_COMBINE_PROMPT_STR)
    elif not isinstance(prompt_template, BasePromptTemplate):
         print(f"⚠️ 指定された prompt_template の型が無効です ({type(prompt_template)})。ChatPromptTemplate を想定しています。")
         prompt_template = ChatPromptTemplate.from_template(DEFAULT_COMBINE_PROMPT_STR) # デフォルトに戻すか警告


    # create_stuff_documents_chain は、文書リストと入力をStuffしてプロンプトに渡し、LLMを呼び出す Runnable を返します。
    combine_chain = create_stuff_documents_chain(llm, prompt_template)

    # Step 2: Retriever と Combine Chain を結合して、Retrieval Chain を構築
    # create_retrieval_chain は、Retriever Runnable と CombineDocumentsChain Runnable を結合し、
    # 入力 {"input": ...} を受け取って、中間結果 (retrieved_documents) と最終結果 (answer) を含む辞書を返す Runnable を構築します。
    retrieval_chain = create_retrieval_chain(retriever, combine_chain)

    return retrieval_chain

def _build_combine_documents_chain(
    llm: Any,
    chain_type: str, # "stuff", "map_reduce", "refine"
    prompt_template: Optional[BasePromptTemplate]
) -> Runnable:
    """
    要約・文書結合チェーンを構築します。stuff / map_reduce / refine の3種に対応。
    これらは基本的に Runnable API に対応した形で構築します。

    Parameters:
    - llm: 言語モデル
    - chain_type: "stuff", "map_reduce", "refine"
    - prompt_template: combine_documents 用のプロンプト（任意）

    Returns:
    - 文書結合チェーン (Runnable)

    Raises:
        ValueError: Unknown combine chain type.
    """
    # 要約・文書結合チェーンは、文書リストを {"context": [...]} の形式で受け取ることを想定
    
    # デフォルトプロンプトを設定
    if prompt_template is None:
        if chain_type == "stuff":
             # Stuffチェーンのデフォルトは質問応答向けが多いが、ここでは要約用も考慮
             # create_stuff_documents_chain はプロンプトなしだとデフォルトを使う
             pass # prompt_template=None のまま create_stuff_documents_chain に渡す
        elif chain_type in ["map_reduce", "refine"]:
             # 要約系のデフォルトプロンプト
             prompt_template = ChatPromptTemplate.from_template("""
             以下の複数文書を日本語で要約してください：

             {context}
             """)
    
    # プロンプトが指定されたが BasePromptTemplate でない場合の警告
    if prompt_template is not None and not isinstance(prompt_template, BasePromptTemplate):
        print(f"⚠️ 指定された prompt_template の型が無効です ({type(prompt_template)})。BasePromptTemplate を想定しています。")
        prompt_template = None # デフォルト動作に戻すかエラーとするか

    if chain_type == "stuff":
        # create_stuff_documents_chain は StuffDocumentsChain (Runnable) を返します
        return create_stuff_documents_chain(llm, prompt_template) # prompt_template=None も許容

    elif chain_type == "refine":
        # create_refine_documents_chain は RefineDocumentsChain (Runnable) を返します
        # refine_documents_chain は通常、initial_prompt と refine_prompt の2つが必要ですが、
        # create_refine_documents_chain は1つのプロンプトからデフォルトの2つを生成するようです。
        # より詳細な制御が必要な場合は、RefineDocumentsChain を直接構築する必要があります。
        if prompt_template is None:
             # RefineChain のデフォルトプロンプトも用意されているので None でも動く
             pass # prompt_template=None のまま渡す
             # あるいは、要約用のデフォルトを明示的に設定
             # prompt_template = ChatPromptTemplate.from_template(...) # 上で設定済み
        return create_refine_documents_chain(llm, prompt_template) # prompt_template=None も許容

    elif chain_type == "map_reduce":
        # MapReduceChain は Runnable として手動構築が必要です (create_* 関数がない)
        # Map と Reduce/Combine で異なるプロンプトを使うのが一般的ですが、
        # 現在の設計では単一の prompt_template を受け取るため、両方に同じプロンプトを使うことになります。
        # これは要約用途であれば適切かもしれません。
        
        if prompt_template is None:
             # MapReduce のデフォルトプロンプトが必要
             # ここでは要約用のデフォルトを設定（上のif文で設定済み）
             prompt_template = ChatPromptTemplate.from_template("""
             以下の複数文書を日本語で要約してください：

             {context}
             """)
        
        if not isinstance(prompt_template, BasePromptTemplate):
             print(f"⚠️ MapReduce チェーン用 prompt_template の型が無効です ({type(prompt_template)})。BasePromptTemplate を想定しています。")
             raise ValueError("MapReduce チェーンには BasePromptTemplate が必要です。")


        # Map チェーン: 各文書チャンクを LLM に渡す
        # LLMChain は Runnable にラップされます (langchain_core.runnables.base.RunnableSequence)
        # ここでは入力変数名を 'context' と想定します。
        map_llm_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Reduce チェーン: Map 結果リストを集約する
        # CombineDocumentsChain として StuffDocumentsChain を使用
        # Mapの結果リストを {"context": [...]} の形式で受け取り、Stuffして llm_chain に渡す
        # ここで llm_chain は最終的なReduce/Combineステップの LLM 呼び出し
        # document_variable_name は、CombineDocumentsChain が入力文書リストを期待する変数名
        combine_document_chain_for_reduce = StuffDocumentsChain(
             llm_chain=LLMChain(llm=llm, prompt=prompt_template), # Reduce/Combine 用の LLMChain
             document_variable_name="context", # MapResult は通常 text のリストなので、context 変数に入れる
             # document_prompt: Mapの結果（文字列）をDocument形式にラップする場合に使うが、ここでは不要
        )

        # ReduceDocumentsChain: Map 結果リストをCombineDocumentChainに渡して集約処理を実行
        reduce_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_document_chain_for_reduce,
            # collapsible_summarize_chain: Optional[Runnable[List[Document], str]] = None # 必要に応じて設定
            # token_max: Optional[int] = None # 必要に応じて設定
        )

        # MapReduceDocumentsChain: Map チェーンと Reduce チェーンを結合
        # Mapチェーンの入力は文書リスト、出力はMap結果リスト
        # Reduceチェーンの入力はMap結果リスト、出力は最終結果
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_llm_chain, # Map フェーズ用の LLMChain (入力は Document, 出力は MapResult)
            reduce_documents_chain=reduce_chain, # Reduce フェーズ用のチェーン
            document_variable_name="context", # MapReduceChain の入力文書リスト変数名
            # process_sef_chain: Optional[Chain] = None # セルフクエリなど高度な機能
        )

        # MapReduceDocumentsChain も Runnable インターフェースを満たします
        return map_reduce_chain

    else:
        # SUPPORTED_CHAINS でフィルタリングしているので、ここには到達しないはず
        raise ValueError(f"Internal error: Unknown chain type passed to _build_combine_documents_chain: {chain_type}")
