# -----------------------------
# 標準ライブラリ
# -----------------------------
import os
import json
from typing import Optional, List, Any, Tuple, Callable, Dict
from pathlib import Path

from dataclasses import dataclass, field

# -----------------------------
# サードパーティライブラリ (LangChain)
# -----------------------------
# Embeddings & VectorStore
from langchain_core.runnables import Runnable # type: ignore
from langchain_community.embeddings import OllamaEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_core.documents import Document # type: ignore


# -----------------------------
# カテゴリ構造定義
# -----------------------------

class RetrieverCategory:
    """
    RAG における文書の分類カテゴリの抽象型。
    サブクラスは分類構造に応じてこの型を継承し、to_dict() および from_dict() を実装する。
    ベクトルストアのメタデータとして保存される形式を定義する。
    """
    def to_dict(self) -> dict:
        """このカテゴリを辞書形式に変換して返す（メタデータ保存用）。"""
        raise NotImplementedError("to_dict() は具象カテゴリで実装してください。")

    @classmethod
    def from_dict(cls, data: dict) -> 'RetrieverCategory':
        """辞書からカテゴリオブジェクトを再構築する（メタデータ読み込み用）。"""
        # 具象クラスで実装が必要
        raise NotImplementedError("from_dict() は具象カテゴリで実装してください。")


@dataclass
class HierarchicalRetrieverCategory(RetrieverCategory):
    """
    階層構造を持つ分類カテゴリ。
    例: {"tagname": "学生", "parent_ids": ["大学", "学部", "理学部"], "level": 3}
    親がいない場合は parent_ids は空のリスト []。
    """
    tagname: str
    parent_ids: List[str] = field(default_factory=list)
    level: int = 0  # 階層レベルを示す整数フィールドを追加。デフォルトは 0。

    def to_dict(self) -> dict:
        # level フィールドも辞書に含める
        return {"tagname": self.tagname, "parent_ids": self.parent_ids, "level": self.level}

    @classmethod
    def from_dict(cls, data: dict) -> 'HierarchicalRetrieverCategory':
        if not isinstance(data, dict):
             raise TypeError("data must be a dictionary")
        return cls(
             tagname=data.get("tagname", ""),
             parent_ids=data.get("parent_ids", []), # デフォルトは空リスト
             level=data.get("level", 0)          # level フィールドを読み込む。デフォルトは 0。
        )

@dataclass
class FlatRetrieverCategory(RetrieverCategory):
    """
    フラット（階層なし）な分類カテゴリ。
    例: {"tagname": "観光"}, {"tagname": "食文化"}
    """
    tagname: str

    def to_dict(self) -> dict:
        return {"tagname": self.tagname}

    @classmethod
    def from_dict(cls, data: dict) -> 'FlatRetrieverCategory':
        if not isinstance(data, dict):
             raise TypeError("data must be a dictionary")
        return cls(tagname=data.get("tagname", ""))

@dataclass
class ArrayRetrieverCategory(RetrieverCategory):
    """
    配列（複数タグ）を持つ分類カテゴリ。
    例: {"tagnames": ["観光", "食文化"]}
    """
    tagnames: list[str]

    def to_dict(self) -> dict:
        # list のコピーを作成して返すことで、元のリストの変更を防ぐ
        return {"tagnames": list(self.tagnames)}

    @classmethod
    def from_dict(cls, data: dict) -> 'ArrayRetrieverCategory':
        if not isinstance(data, dict):
             raise TypeError("data must be a dictionary")
        tagnames = data.get("tagnames", [])
        if not isinstance(tagnames, list):
             print(f"⚠️ 'tagnames' はリストである必要がありますが、{type(tagnames)} です。空リストとして処理します。")
             tagnames = []
        return cls(tagnames=tagnames)

# メタデータからカテゴリを再構築するためのファクトリ関数
def category_from_dict(data: dict) -> Optional[RetrieverCategory]:
    """
    メタデータ辞書から適切な RetrieverCategory オブジェクトを再構築する。
    辞書の構造を見て、どのカテゴリクラスで復元すべきかを判断する。
    """
    if not isinstance(data, dict):
        return None

    # 例えば、tagname キーがあれば Flat または Hierarchical と判断
    if "tagname" in data:
        # parent_ids があれば Hierarchical と判断
        if "parent_ids" in data and isinstance(data["parent_ids"], list):
            return HierarchicalRetrieverCategory.from_dict(data)
        else:
            return FlatRetrieverCategory.from_dict(data)
    # tagnames キーがあれば Array と判断
    elif "tagnames" in data and isinstance(data["tagnames"], list):
        return ArrayRetrieverCategory.from_dict(data)
    else:
        # どのカテゴリ構造にも一致しない場合
        print(f"⚠️ 未知のカテゴリ構造が見つかりました: {data}")
        return None

# -----------------------------
# Retriever および VectorStore 操作関数
# -----------------------------
def create_faiss_retriever(
    vectorstore: FAISS,
    k: int = 5,
    score_threshold: Optional[float] = None # スコア閾値は Retriever 検索時にフィルタリングで利用可能
) -> Runnable: # Retriever オブジェクトは Runnable です
    """
    FAISS ベクトルストアから Runnable な Retriever を作成する。

    Parameters:
    - vectorstore: FAISS オブジェクト
    - k: 検索文書数
    - score_threshold: 検索結果をフィルタリングするスコアしきい値（任意）

    Returns:
    - retriever: LangChain Retriever オブジェクト (Runnable)
    """
    # FAISS.as_retriever はキーワード引数 search_kwargs を通じて検索パラメータを設定できる
    search_kwargs = {"k": k}

    # スコア閾値フィルタリングは検索結果に対して後処理で行う必要があることが多い。
    # as_retriever に score_threshold を直接設定する標準的な方法は LangChain にはないため、
    # k のみ search_kwargs に渡し、スコア閾値は Retriever 実行後に手動でフィルタリングするか、
    # カスタム Retriever ラッパーを使用する必要があります。
    # ここでは score_threshold 引数は docstring に残しつつ、as_retriever には渡しません。

    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def load_faiss_retriever(vect_path: str, k: int = 5) -> Optional[Runnable]:
    """
    FAISS ベクトルストアをロードし、Runnable な Retriever を返します。

    Parameters:
    - vect_path: ベクトルストアの保存パス
    - k: 検索文書数

    Returns:
    - Retriever オブジェクト (Runnable)（成功時）、または None（失敗時）
    """
    vectorstore = load_vectorstore(vect_path)
    if vectorstore is not None:
        # 関数名変更: create_retriever -> create_faiss_retriever
        return create_faiss_retriever(vectorstore, k=k)
    else:
        return None

def load_vectorstore(vect_path: str) -> Optional[FAISS]:
    """
    FAISS ベクトルストアをローカルパスからロードします。

    Parameters:
    - vect_path: ベクトルストアディレクトリのパス

    Returns:
    - FAISS Vectorstore オブジェクト（成功時）、または None（失敗時）
    """
    vect_dir = Path(vect_path)

    if not vect_dir.exists() or not vect_dir.is_dir():
         print(f"エラー: ベクトルストアディレクトリが存在しません: {vect_path}")
         return None

    index_file = vect_dir / "index.faiss"
    if not index_file.exists():
        print(f"エラー: ベクトルストアインデックスファイルが見つかりません: {index_file}")
        return None

    metadata_path = vect_dir / "metadata.json"
    if not metadata_path.exists():
         print(f"エラー: メタデータファイルが見つかりません: {metadata_path}")
         # メタデータは必須情報（特に埋め込みモデル名）を含むため、メタデータがない場合はロード失敗とする
         return None

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        embedding_model = metadata.get("embedding_model")
        if not embedding_model:
             print(f"エラー: メタデータに 'embedding_model' が指定されていません: {metadata_path}")
             return None

        # OllamaEmbeddings インスタンスを作成
        try:
            embeddings = OllamaEmbeddings(model=embedding_model) # type: ignore
        except Exception as e:
            print(f"エラー: 埋め込みモデル '{embedding_model}' の初期化に失敗しました: {e}")
            return None

        # FAISS.load_local に allow_dangerous_deserialization=True が必要 (セキュリティリスクを理解した上で使用)
        # FAISS >= 1.0.0 から必須になりました。信頼できるソースからのみロードしてください。
        vectorstore = FAISS.load_local(str(vect_dir), embeddings, allow_dangerous_deserialization=True) # type: ignore
        print(f"✅ ベクトルストアをロードしました: {vect_path}")
        return vectorstore

    except FileNotFoundError: # 上記でチェック済みだが念のため
         print(f"エラー: ベクトルストアファイルまたはメタデータファイルが見つかりません。")
         return None
    except json.JSONDecodeError:
         print(f"エラー: メタデータファイルのJSON形式が不正です: {metadata_path}")
         return None
    except Exception as e:
        print(f"❌ ベクトルストアのロード中に予期せぬエラーが発生しました: {e}")
        return None

def merge_vectorstore(vect_paths: list[str]) -> FAISS:
    """
    複数の FAISS ベクトルストアを統合し、新しい Vectorstore を構築する。
    統合されたベクトルストアは、元のドキュメントの UUID を保持しないため、
    再構築時に新しい UUID が割り当てられる（from_documentsで再生成されるため）。

    Parameters:
    - vect_paths: ベクトルストアディレクトリのリスト

    Returns:
    - 統合された FAISS vectorstore オブジェクト

    Raises:
    - ValueError: ベクトルストアパスリストが空、埋め込みモデルの不一致、または統合対象の文書がない場合
    - RuntimeError: 処理中に予期せぬエラーが発生した場合
    """
    if not vect_paths:
        raise ValueError("ベクトルストアパスリストが空です。統合できません。")

    print("[DEBUG] 統合対象パス:")
    for p in vect_paths:
        print("  -", p)

    all_docs = []
    embedding_models = set()

    for path in vect_paths:
        try:
            store = load_vectorstore(path)
            if store is None:
                print(f"⚠️ ベクトルストアの読み込みに失敗したため、スキップします: {path}")
                continue

            # VectorStore から Document オブジェクトを取得
            # FAISS の docstore は内部的には dict のような構造で Document オブジェクトを保持
            docs = list(store.docstore._dict.values())

            # 統合元のメタデータから埋め込みモデル名を取得 (再構築時に必要)
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                 try:
                     with open(metadata_path, encoding="utf-8") as f:
                         metadata = json.load(f)
                         model_name = metadata.get("embedding_model")
                         if model_name:
                             embedding_models.add(model_name)
                         else:
                             print(f"⚠️ メタデータファイル '{metadata_path}' に 'embedding_model' がありません。")
                 except Exception as e:
                     print(f"⚠️ メタデータファイル '{metadata_path}' の読み込みまたは解析に失敗しました: {e}")
            else:
                 print(f"⚠️ メタデータファイルが見つかりません: {metadata_path}")

            # ロードしたドキュメントリストに追加
            all_docs.extend(docs)

        except Exception as e:
            print(f"⚠️ ベクトルストア '{path}' の処理中にエラーが発生しました: {e}")


    if not all_docs:
        raise ValueError("統合対象の文書が一つも見つかりませんでした。")

    if len(embedding_models) == 0:
         raise ValueError("統合対象のベクトルストアから埋め込みモデル名を特定できませんでした。")
    elif len(embedding_models) > 1:
        raise ValueError(f"複数の埋め込みモデルが混在しています: {embedding_models}")

    embedding_model = embedding_models.pop()

    try:
        embeddings = OllamaEmbeddings(model=embedding_model) # type: ignore
    except Exception as e:
         raise RuntimeError(f"埋め込みモデル '{embedding_model}' の初期化に失敗しました: {e}") from e


    print(f"✅ {len(all_docs)} 件の文書を使用してベクトルストアを再構築します（モデル: {embedding_model}）")
    try:
        # from_documents で新しい FAISS インデックスを作成
        merged_vectorstore = FAISS.from_documents(all_docs, embeddings) # type: ignore
        return merged_vectorstore
    except Exception as e:
         raise RuntimeError(f"FAISS ベクトルストアの再構築に失敗しました: {e}") from e

# -----------------------------
# ベクトルストア メタデータ編集関数 (カテゴリ関連)
# -----------------------------
def edit_vectorstore_metadata(
    vectorstore: FAISS,
    edit_function: Callable[[Document], Dict[str, Any]]
) -> FAISS:
    """
    FAISS ベクトルストア内のすべての文書のメタデータを編集します。
    元のベクトルストアオブジェクトがインプレースで更新されます。

    Parameters:
    - vectorstore: 編集対象の FAISS ベクトルストア
    - edit_function: 各 Document (langchain_core.documents.Document) を受け取り、
                     更新後の metadata 辞書 (dict) を返す関数。
                     この関数は、新しいメタデータ全体を返す必要があります。

    Returns:
    - 編集済みのベクトルストア（元のオブジェクトへの参照）

    Raises:
    - ValueError: edit_function が dict を返さなかった場合
    """
    # FAISS の docstore は内部的には dict のような構造で Document オブジェクトを保持
    docs = vectorstore.docstore._dict.values()

    print(f"📝 ベクトルストア内の {len(docs)} 件の文書メタデータを編集します...")
    for doc in docs:
        # edit_function を呼び出して新しいメタデータを取得
        new_metadata = edit_function(doc)

        if isinstance(new_metadata, dict):
            # メタデータを更新
            doc.metadata = new_metadata
        else:
            # edit_function が無効な値を返した場合
            raise ValueError(f"edit_function は新しい metadata 辞書を返す必要がありますが、{type(new_metadata)} が返されました")

    print(f"✅ {len(docs)} 件の文書メタデータの編集が完了しました。")
    return vectorstore

# RetrieverCategory を使用した具体的なメタデータ編集ヘルパー関数群 (ファクトリ関数)

def make_category_editor(new_category: RetrieverCategory) -> Callable[[Document], Dict[str, Any]]:
    """
    RetrieverCategory を用いて、既存の 'category' メタデータに部分的に値を上書き（マージ）する
    編集関数を生成して返します。

    Parameters:
    - new_category: ドキュメントの 'category' メタデータにマージしたい値を持つ RetrieverCategory のインスタンス

    Returns:
    - Callable[[Document], dict]: edit_vectorstore_metadata に渡すための編集関数
    """
    new_category_dict = new_category.to_dict() # 事前に dict に変換

    def editor(doc: Document) -> Dict[str, Any]:
        # Document オブジェクトの metadata は dict です
        metadata = dict(doc.metadata) # 元のメタデータをコピー

        # 既存の category メタデータを取得。dict でなければ初期化
        category = metadata.get("category", {})
        if not isinstance(category, dict):
            print(f"⚠️ Document '{doc.metadata.get('doc_id', '不明')}' の既存カテゴリメタデータが無効です ({type(category)})。上書きします。")
            category = {}

        # 新しいカテゴリ値を既存のカテゴリ辞書にマージ
        # 新しいカテゴリに含まれるキーが既存カテゴリにあれば上書きされます
        category.update(new_category_dict)

        # 更新した category 辞書をメタデータに戻す
        metadata["category"] = category

        return metadata # 更新後のメタデータ全体を返す

    return editor

def make_category_remover(removal_category_keys: list[str]) -> Callable[[Document], Dict[str, Any]]:
    """
    指定されたキーをドキュメントの 'category' メタデータから削除する編集関数を生成して返します。

    Parameters:
    - removal_category_keys: 'category' 辞書から削除したいキーのリスト

    Returns:
    - Callable[[Document], dict]: edit_vectorstore_metadata に渡すための編集関数
    """
    def editor(doc: Document) -> Dict[str, Any]:
        metadata = dict(doc.metadata)
        category = metadata.get("category", {})

        if isinstance(category, dict):
            for key in removal_category_keys:
                category.pop(key, None) # キーが存在しなくてもエラーにならない
            metadata["category"] = category # 削除後のcategoryを戻す
        else:
             print(f"⚠️ Document '{doc.metadata.get('doc_id', '不明')}' の既存カテゴリメタデータが無効です ({type(category)})。削除処理をスキップします。")

        return metadata

    return editor

def make_category_replacer(new_category: RetrieverCategory) -> Callable[[Document], Dict[str, Any]]:
    """
    RetrieverCategory を用いて、ドキュメントの 'category' メタデータを完全に置き換える
    編集関数を生成して返します。

    Parameters:
    - new_category: ドキュメントの 'category' メタデータとして設定したい RetrieverCategory のインスタンス

    Returns:
    - Callable[[Document], dict]: edit_vectorstore_metadata に渡すための編集関数
    """
    new_category_dict = new_category.to_dict() # 事前に dict に変換

    def editor(doc: Document) -> Dict[str, Any]:
        metadata = dict(doc.metadata)
        # 既存の category を完全に新しい category 辞書で置き換える
        metadata["category"] = new_category_dict
        return metadata

    return editor