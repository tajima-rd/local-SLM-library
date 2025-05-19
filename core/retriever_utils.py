import os
import json
from uuid import uuid4

from langchain_ollama import OllamaEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore

from dataclasses import dataclass

class RetrieverCategory:
    """
    RAG における分類カテゴリの抽象型。
    サブクラスは分類構造に応じてこの型を継承する。
    """

    def to_dict(self) -> dict:
        raise NotImplementedError("to_dict() は具象カテゴリで実装してください。")

@dataclass
class FlatRetrieverCategory(RetrieverCategory):
    """
    フラット（階層なし）な分類カテゴリ。
    例: {"tagname": "観光"}, {"tagname": "食文化"}
    """
    tagname: str

    def to_dict(self) -> dict:
        return {"tagname": self.tagname}

def create_retriever(vectorstore, k: int = 5, score_threshold: float = None):
    """
    ベクトルストアから Retriever を作成する。

    Parameters:
    - vectorstore: FAISS オブジェクト
    - k: 検索文書数
    - score_threshold: スコアしきい値（任意）

    Returns:
    - retriever: LangChain Retriever オブジェクト
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    if score_threshold is not None:
        retriever.score_threshold = score_threshold  # カスタム属性として使用可能
    return retriever

def load_vectorstore(vect_path: str):
    """
    FAISS ベクトルストアをローカルパスからロードします。

    Parameters:
    - vect_path: ベクトルストアディレクトリのパス

    Returns:
    - Vectorstore オブジェクト（成功時）、または None（失敗時）
    """
    try:
        metadata_path = os.path.join(vect_path, "metadata.json")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        embedding_model = metadata["embedding_model"]
        embeddings = OllamaEmbeddings(model=embedding_model)

        vectorstore = FAISS.load_local(vect_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    except Exception as e:
        print(f"ベクトルストアのロードに失敗しました: {e}")
        return None

def load_retriever(vect_path: str, k: int = 5):
    """
    ベクトルストアをロードし、Retriever を返します。

    Parameters:
    - vect_path: ベクトルストアの保存パス
    - k: 検索文書数

    Returns:
    - Retriever オブジェクト（成功時）、または None（失敗時）
    """
    vectorstore = load_vectorstore(vect_path)
    if vectorstore is not None:
        return create_retriever(vectorstore, k=k)
    else:
        return None

def merge_vectorstore(vect_paths: list[str]):
    """
    複数の FAISS ベクトルストアを統合し、新しい Vectorstore を構築する。

    Parameters:
    - vect_paths: ベクトルストアディレクトリのリスト

    Returns:
    - FAISS vectorstore オブジェクト

    Raises:
    - ValueError: 埋め込みモデルの不一致、またはベクトルストアが空の場合
    """
    if not vect_paths:
        raise ValueError("ベクトルストアパスリストが空です。")

    print("[DEBUG] 統合対象パス:")
    for p in vect_paths:
        print("  -", p)

    all_docs = []
    embedding_models = set()

    for path in vect_paths:
        store = load_vectorstore(path)
        if store is None:
            print(f"⚠️ ベクトルストアの読み込みに失敗: {path}")
            continue

        docs = list(store.docstore._dict.values())
        for doc in docs:
            doc.metadata["doc_id"] = str(uuid4())  # UUID 再割当て
        all_docs.extend(docs)

        metadata_path = os.path.join(path, "metadata.json")
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
                embedding_models.add(metadata["embedding_model"])
        except Exception as e:
            print(f"⚠️ メタデータ読み込みに失敗: {metadata_path}\n{e}")

    if not all_docs:
        raise RuntimeError("統合対象の文書が存在しません。")

    if len(embedding_models) != 1:
        raise ValueError(f"複数の埋め込みモデルが混在しています: {embedding_models}")

    embedding_model = embedding_models.pop()
    embeddings = OllamaEmbeddings(model=embedding_model)

    print(f"✅ {len(all_docs)} 件の文書を使用してベクトルストアを再構築します（モデル: {embedding_model}）")
    return FAISS.from_documents(all_docs, embeddings)

def edit_vectorstore_metadata(vectorstore, edit_function):
    """
    FAISS ベクトルストア内のすべての文書メタデータを編集します。

    Parameters:
    - vectorstore: 編集対象の FAISS ベクトルストア
    - edit_function: 各 Document を受け取り、更新後の metadata を返す関数

    Returns:
    - 編集済みのベクトルストア（in-place 更新）
    
    Raises:
    - ValueError: edit_function が dict を返さなかった場合
    """
    docs = vectorstore.docstore._dict.values()

    for doc in docs:
        new_metadata = edit_function(doc)
        if isinstance(new_metadata, dict):
            doc.metadata = new_metadata
        else:
            raise ValueError("edit_function は新しい metadata 辞書を返す必要があります")

    return vectorstore

def make_category_editor(new_category: RetrieverCategory):
    """
    RetrieverCategory を用いて、既存の category に部分的に値を上書きする。

    Parameters:
    - new_category: RetrieverCategory のインスタンス

    Returns:
    - Callable[[Document], dict]
    """
    def editor(doc):
        metadata = dict(doc.metadata)
        category = metadata.get("category", {})

        if not isinstance(category, dict):
            category = {}

        category.update(new_category.to_dict())
        metadata["category"] = category
        return metadata
    return editor


def make_category_adder(new_category: RetrieverCategory):
    """
    RetrieverCategory を用いて category を dict としてマージする。

    Parameters:
    - new_category: RetrieverCategory のインスタンス

    Returns:
    - Callable[[Document], dict]
    """
    def editor(doc):
        metadata = dict(doc.metadata)
        category = metadata.get("category", {})

        if not isinstance(category, dict):
            category = {}

        category.update(new_category.to_dict())
        metadata["category"] = category
        return metadata
    return editor

def make_category_remover(removal_category: RetrieverCategory):
    """
    RetrieverCategory が保持する key を category から削除する編集関数を返す。

    Parameters:
    - removal_category: 削除対象のキーを含む RetrieverCategory インスタンス

    Returns:
    - Callable[[Document], dict]
    """
    def editor(doc):
        metadata = dict(doc.metadata)
        category = metadata.get("category", {})

        if isinstance(category, dict):
            for key in removal_category.to_dict().keys():
                category.pop(key, None)
            metadata["category"] = category

        return metadata
    return editor


def make_category_replacer(new_category: RetrieverCategory):
    """
    RetrieverCategory を用いて category を完全に置き換える編集関数を返す。

    Parameters:
    - new_category: RetrieverCategory のインスタンス

    Returns:
    - Callable[[Document], dict]
    """
    def editor(doc):
        metadata = dict(doc.metadata)
        metadata["category"] = new_category.to_dict()
        return metadata
    return editor

