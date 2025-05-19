import os, json, shutil
from uuid import uuid4
from pathlib import Path

from langchain_ollama import OllamaEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.docstore.document import Document


from retriever_utils import edit_vectorstore_metadata  # メタ編集が必要な場合
from retriever_utils import RetrieverCategory  # ✅ 追加
import document_utils as docutils  # 別モジュールから文書処理関数を呼び出す想定

def vectorization(
    md_path: str,
    vect_path: str,
    category: RetrieverCategory,
    overwrite: bool = False,
    embedding_name: str = "nomic-embed-text:latest"
):
    """
    入力ファイルを処理してベクトルストアを構築・保存する。

    Parameters:
    - in_file: 入力ファイルパス（PDF, DOCX, MD など）
    - md_path: 変換後のMarkdown出力パス
    - vect_path: 保存先ベクトルストアパス
    - category: 文書に付与するカテゴリ（metadata に記録）
    - overwrite: True の場合は既存ベクトルストアを上書き
    - embedding_name: 使用する埋め込みモデル名

    Returns:
    - FAISS vectorstore オブジェクト、または None（失敗時）
    """
    print("※ granite-embedding:278m を選ぶと処理が落ちます…")

    # 埋め込みモデルの初期化
    embeddings = OllamaEmbeddings(model=embedding_name)

    # 入力形式の確認と Markdown 変換
    doc_path = Path(md_path)
    doc_format = docutils.get_document_format(doc_path)

    if not doc_format:
        print(f"サポートされていないドキュメント形式: {doc_path.suffix}")
        return None

    if not doc_format.name == "MD":
        return None

    # 上書きフラグまたは非存在時にベクトルストアを作成
    if overwrite or not os.path.exists(vect_path):
        print("ベクトルストアを保存します。")
        success = save_chain(
            md_path=md_path,
            vect_path=vect_path,
            embeddings=embeddings,
            category=category,
            loader_type="markdown",
            text_splitter_type="recursive"
        )
        return success
    else:
        print(f"既存のベクトルストアが存在します: {vect_path}（上書きしません）")
        return None

def save_chain(
    md_path: str,
    vect_path: str,
    embeddings: OllamaEmbeddings,
    category: RetrieverCategory,
    loader_type: str = "markdown",
):
    
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