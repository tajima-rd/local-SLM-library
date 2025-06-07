# -----------------------------
# 標準ライブラリ
# -----------------------------
import os
import json
from uuid import uuid4
from typing import List, Any, Optional, Dict
from pathlib import Path

# -----------------------------
# サードパーティライブラリ (LangChain)
# -----------------------------
from langchain_community.embeddings import OllamaEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_core.documents import Document # type: ignore
from langchain_core.runnables import Runnable # type: ignore
from langchain_core.prompts import BasePromptTemplate # type: ignore # チェーン構築関数で必要になる可能性


# -----------------------------
# 自作モジュール
# -----------------------------
from . import document_utils as docutils
from . import retriever_utils # RetrieverCategory, load_vectorstore, create_faiss_retriever, merge_vectorstore などが必要
from .retriever_utils import RetrieverCategory, load_vectorstore, merge_vectorstore, create_faiss_retriever

try:
    from .chain_factory import get_chain # 相対インポートに修正
except ImportError:
    print("⚠️ chain_factory モジュールが見つかりません。 prepare_chain_from_path 関数は使用できません。")
    get_chain = None # インポートできなかった場合は None に設定

def _save_documents_to_faiss(
    documents: List[Document],
    vect_path: str,
    embedding_name: str,
    category: retriever_utils.RetrieverCategory,
    loader_type: str, # loader_type はメタデータに保存用
) -> bool:
    """
    文書リストを受け取り、分割、メタデータ付与、FAISS保存を行うプライベートヘルパー関数。

    Parameters:
    - documents: 保存対象の LangChain Document オブジェクトのリスト
    - vect_path: ベクトルストアを保存するディレクトリのパス
    - embedding_name: 使用する埋め込みモデルの名前 (Ollama モデル名など)
    - category: このドキュメントバッチに紐づける RetrieverCategory オブジェクト
    - loader_type: ドキュメントの元となったローダーのタイプを示す文字列 (例: "markdown", "text")

    Returns:
    - bool: 保存に成功したか
    """
    # 埋め込みモデルの初期化
    try:
        embeddings = OllamaEmbeddings(model=embedding_name) # type: ignore
    except Exception as e:
        print(f"❌ 埋め込みモデルの初期化に失敗しました ({embedding_name}): {e}")
        return False

    print("テキストを分割してチャンクを割り当てます。")
    try:
        # ロード元の loader_type に基づいて適切なスプリッターを推奨
        splitter = docutils.suggest_text_splitter(
            documents=documents,
            loader_type=loader_type
        )
        split_docs = splitter.split_documents(documents)
    except Exception as e:
        print(f"❌ テキスト分割中にエラーが発生しました: {e}")
        return False


    # 各チャンクにメタデータ（doc_id, categoryなど）を付与
    for doc in split_docs:
        # 既存の doc_id がある場合もあるが、再構築時は新しい UUID を割り当てるのが安全
        doc.metadata["doc_id"] = str(uuid4())

        # category メタデータの更新/追加 (既存があればマージ)
        # ドキュメント自身のメタデータに category 情報を持たせる
        existing_category = doc.metadata.get("category", {})
        if not isinstance(existing_category, dict):
             existing_category = {} # 既存がdictでない場合は初期化

        # category を辞書に変換してマージ
        # 新しいカテゴリに含まれるキーが既存カテゴリにあれば上書きされます
        doc.metadata["category"] = {**existing_category, **category.to_dict()}

    # チャンクがない場合は保存しない
    if not split_docs:
         print("⚠️ 分割されたドキュメントがありません。ベクトルストアは保存されません。")
         return False

    # FAISS保存
    print(f"✨ {len(split_docs)} 件のチャンクをベクトルストアに保存します。")
    if split_docs:
         # 最初のチャンクの内容とメタデータをデバッグ表示
         print(f"[DEBUG] 最初のチャンク (一部): {split_docs[0].page_content[:200]}...")
         print(f"[DEBUG] 最初のチャンク メタデータ: {split_docs[0].metadata}")

    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings) # type: ignore

        # ディレクトリが存在しない場合があるので作成
        Path(vect_path).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(vect_path)

    except Exception as e:
        print(f"❌ FAISS ベクトルストアの保存に失敗しました: {e}")
        return False


    # メタデータの保存 (ベクトルストアディレクトリ全体に関する情報)
    metadata = {
        "embedding_model": embedding_name, # モデル名自体を保存
        "loader_type": loader_type,
        "category": category.to_dict(), # この保存バッチ全体の代表カテゴリとして記録
        # その他、スプリッター設定などを追加するのも良いかもしれない
    }
    metadata_file_path = os.path.join(vect_path, "metadata.json")
    try:
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ メタデータファイル ({metadata_file_path}) の保存に失敗しました: {e}")
        # ベクトルストア自体は保存できている可能性があるので True を返すことも検討
        # ここでは一貫性のため False としますが、設計による

    print(f"✅ ベクトルストアとメタデータを保存しました: {vect_path}")
    return True

def save_markdown_to_vectorstore(
    md_path: str,
    vect_path: str,
    embedding_name: str,
    category: RetrieverCategory,
    loader_type: str = "markdown",
) -> bool:
    """
    Markdownファイルからドキュメントをロード、分割し、FAISSベクトルストアとして保存する。

    Parameters:
    - md_path: 入力Markdownファイルのパス
    - vect_path: ベクトルストアを保存するディレクトリのパス
    - embedding_name: 使用する埋め込みモデル名
    - category: このファイルに紐づける RetrieverCategory オブジェクト
    - loader_type: 使用するローダータイプ (デフォルト: "markdown")

    Returns:
    - bool: 保存に成功したか
    """
    if not os.path.exists(md_path):
        print(f"❌ 入力ファイルが見つかりません: {md_path}")
        return False

    try:
        # ドキュメントのロード
        documents = docutils.load_documents(md_path, loader_type)
        if not documents:
             print(f"⚠️ ファイル {md_path} からドキュメントをロードできませんでした。")
             return False

        # ヘルパー関数を呼び出して保存
        return _save_documents_to_faiss(
            documents=documents,
            vect_path=vect_path,
            embedding_name=embedding_name,
            category=category,
            loader_type=loader_type
        )
    except Exception as e:
        print(f"❌ Markdownからの保存処理中にエラーが発生しました: {e}")
        return False

def save_text_to_vectorstore(
    text: str,
    vect_path: str,
    embedding_name: str,
    category: RetrieverCategory,
    loader_type: str = "text",
) -> bool:
    """
    プレーンテキスト文字列からドキュメントを生成し、分割、FAISSベクトルストアとして保存する。

    Parameters:
    - text: 保存対象のプレーンテキスト文字列
    - vect_path: ベクトルストアを保存するディレクトリのパス
    - embedding_name: 使用する埋め込みモデル名
    - category: このテキストに紐づける RetrieverCategory オブジェクト
    - loader_type: 使用するローダータイプ (デフォルト: "text")

    Returns:
    - bool: 保存に成功したか
    """
    if not text:
        print("⚠️ 保存対象のテキストが空です。")
        return False

    try:
        # Documentに変換（metadataはあとで付与）
        document = Document(page_content=text, metadata={})
        documents = [document]

        # ヘルパー関数を呼び出して保存
        return _save_documents_to_faiss(
            documents=documents,
            vect_path=vect_path,
            embedding_name=embedding_name,
            category=category,
            loader_type=loader_type
        )
    except Exception as e:
        print(f"❌ テキストからの保存処理中にエラーが発生しました: {e}")
        return False

def prepare_chain_from_path(
    llm: Any,
    faiss_paths: list[Path],
    chain_type: str = "conversational",
    k: int = 5,
    prompt_template: Optional[BasePromptTemplate] = None,
    **kwargs: Any # get_chain に渡す追加引数
) -> Optional[Runnable]:
    """
    指定パス配下の FAISS ベクトルストアを統合し、Retriever および RAG チェーンを構築する。
    (Ingestion + Chain Factory の組み合わせ機能)

    Parameters:
    - llm: 言語モデル
    - faiss_paths: ベクトルストア (.faiss および index.faiss を含むディレクトリ) のパスのリスト
    - chain_type: チェーンの種類 ("conversational", "retrievalqa" など)
    - k: 検索数（Retriever用）
    - prompt_template: combine_documents 用のプロンプト（任意）
    - kwargs: get_chain に渡すその他のオプション

    Returns:
    - LangChain チェーンオブジェクト (Runnable)（成功時）、または None（失敗時）

    Raises:
    - FileNotFoundError: ベクトルストアパスリストが空の場合、またはパスが存在しない場合。
    - ValueError: ベクトルストアの統合に問題がある場合 (例: モデル不一致)。
    - Exception: チェーン構築中のエラー。
    """
    if get_chain is None:
        print("❌ chain_factory モジュールがロードされていないため、prepare_chain_from_path は実行できません。")
        return None

    if not faiss_paths:
        raise FileNotFoundError("ベクトルストア（.faiss ディレクトリ）のパスが指定されていません。")

    print(f"🔍 {len(faiss_paths)} 件のベクトルストアを統合します。")

    # パスが実際に存在するかチェック
    existing_paths = [p for p in faiss_paths if p.exists() and p.is_dir()]
    if len(existing_paths) != len(faiss_paths):
         missing_paths = set(faiss_paths) - set(existing_paths)
         print(f"⚠️ 指定されたベクトルストアパスの一部または全てが見つかりません: {[str(p) for p in missing_paths]}")
         # 見つかったものだけでも処理を試みるか、エラーとするか
         # ここではエラーとします
         raise FileNotFoundError(f"指定されたベクトルストアパスが見つかりません: {[str(p) for p in missing_paths]}")

    try:
        # ベクトルストアを統合して Retriever を作成 (retriever_utils の機能を使用)
        vectorstore = merge_vectorstore([str(p) for p in existing_paths])
        # 関数名変更: create_retriever -> create_faiss_retriever
        retriever = create_faiss_retriever(vectorstore, k=k)
    except Exception as e:
         print(f"❌ ベクトルストアの統合またはRetrieverの作成に失敗しました: {e}")
         return None

    # 統合されたRetrieverを使ってチェーンを作成 (chain_factory の機能を使用)
    try:
        chain = get_chain(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            prompt_template=prompt_template,
            k=k, # k は get_chain に渡す (会話履歴ありの場合の retriever にも影響)
            **kwargs, # その他の引数も渡す
        )
        print(f"✅ {chain_type} チェーンを構築しました。")
        return chain
    except Exception as e:
        print(f"❌ チェーン構築に失敗しました ({chain_type}): {e}")
        # traceback.print_exc() # 呼び出し元で出力されるべき
        return None