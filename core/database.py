import sqlite3, os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field

from . import document_utils
from . import retriever_utils
from .rag_session import RAGSession # type: ignore # type: ignoreはそのまま残しておく

# === 抽象クラス定義 ===

from abc import ABC, abstractmethod

class DBObject(ABC):
    @classmethod
    @abstractmethod
    def table_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_row(cls, row: Tuple[Any]) -> 'DBObject':
        pass

    @abstractmethod
    def insert(self, conn: sqlite3.Connection) -> int:
        pass

    @abstractmethod
    def update(self, conn: sqlite3.Connection):
        pass

    @abstractmethod
    def delete(self, conn: sqlite3.Connection):
        pass

# === データモデル定義 ===
@dataclass(init=False)
class Category(DBObject):
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    parent_ids: List[int] = field(default_factory=list) # 新規追加
    type_code: str = "hier"
    sort_order: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        parent_ids: Optional[List[int]] = None, # 新規追加
        type_code: str = "hier",
        sort_order: int = 0,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.name = name
        self.description = description
        self.parent_ids = parent_ids if parent_ids is not None else [] # 新規追加
        self.type_code = type_code
        self.sort_order = sort_order
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            # insert メソッド自体が中間テーブルへの挿入も行う
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "categories"
    
    def from_row(cls, row: Tuple[Any]) -> 'Category':
        # 既存のタプル構造にparent_idが含まれていない前提で調整
        # from_row で使用するタプルの順序とカラムを再確認する必要がある
        # DBからSELECT *した場合の順序に合わせて調整
        # SELECT id, name, description, type_code, sort_order, created_at, updated_at
        # 修正: parent_id をタプルから読み取らない
        # 元: return cls(*row) # id, name, description, parent_id, type_code, sort_order, created_at, updated_at の順だったはず
        # 新: id, name, description, type_code, sort_order, created_at, updated_at
        if len(row) != 7: # id, name, description, type_code, sort_order, created_at, updated_at の7カラムを想定
             raise ValueError(f"Expected 7 columns for Category.from_row, got {len(row)}")

        cat = cls.__new__(cls) # __init__ をスキップ
        cat.id = row[0]
        cat.name = row[1]
        cat.description = row[2]
        cat.parent_ids = [] # from_row 時点では親情報はロードしない
        cat.type_code = row[3]
        cat.sort_order = row[4]
        cat.created_at = row[5]
        cat.updated_at = row[6]
        return cat
    
    @classmethod
    def get_all_categories(cls, conn: sqlite3.Connection) -> list["Category"]:
        cur = conn.cursor()
        # 親情報なしで基本のカテゴリ情報を取得
        cur.execute("""
            SELECT id, name, description, type_code, sort_order, created_at, updated_at
            FROM categories
        """)
        category_rows = cur.fetchall()

        # 中間テーブルから全ての親子関係を取得
        cur.execute("""
            SELECT child_category_id, parent_category_id
            FROM category_parents
        """)
        parent_links = cur.fetchall()

        # 親リンク情報を、子カテゴリIDをキーとする辞書に整理
        parent_map = {}
        for child_id, parent_id in parent_links:
            if child_id not in parent_map:
                parent_map[child_id] = []
            parent_map[child_id].append(parent_id)

        # Category オブジェクトを生成し、親リンク情報を設定
        categories = []
        for row in category_rows:
            cat = cls.from_row(row) # 基本情報のみ設定 (parent_ids は [] で初期化されている)
            if cat.id in parent_map:
                cat.parent_ids = parent_map[cat.id] # 該当する親IDリストを設定
            categories.append(cat)

        return categories

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        # categories テーブルへの挿入 (parent_id カラムなし)
        cur.execute('''
            INSERT INTO categories (name, description, type_code, sort_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        ''', (self.name, self.description, self.type_code, self.sort_order))
        
        self.id = cur.lastrowid # 挿入されたカテゴリのIDを取得

        # 中間テーブル category_parents への挿入
        if self.parent_ids:
            # 重複を避けるために set を使うか、または一括挿入
            links_to_insert = [(self.id, parent_id) for parent_id in set(self.parent_ids) if parent_id is not None]
            if links_to_insert:
                 cur.executemany('''
                     INSERT INTO category_parents (child_category_id, parent_category_id)
                     VALUES (?, ?)
                 ''', links_to_insert)

        conn.commit()
        return self.id

    def update(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないCategoryオブジェクトは更新できません。")

        cur = conn.cursor()
        # categories テーブルの更新 (parent_id カラムなし)
        cur.execute('''
            UPDATE categories SET name=?, description=?, type_code=?, sort_order=?, updated_at=datetime('now')
            WHERE id=?
        ''', (self.name, self.description, self.type_code, self.sort_order, self.id))
        
        # 中間テーブル category_parents の更新
        # 既存の親リンクを全て削除
        cur.execute('DELETE FROM category_parents WHERE child_category_id=?', (self.id,))
        
        # 新しい親リンクを挿入
        if self.parent_ids:
            links_to_insert = [(self.id, parent_id) for parent_id in set(self.parent_ids) if parent_id is not None]
            if links_to_insert:
                cur.executemany('''
                    INSERT INTO category_parents (child_category_id, parent_category_id)
                    VALUES (?, ?)
                ''', links_to_insert)

        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないCategoryオブジェクトは削除できません。")
        cur = conn.cursor()
        # categories テーブルから削除 -> 中間テーブルの対応するエントリは CASCADE で削除される
        cur.execute('DELETE FROM categories WHERE id=?', (self.id,))
        conn.commit()

    @classmethod
    def get_by_id(cls, conn: sqlite3.Connection, category_id: int) -> Optional["Category"]:
        """
        指定されたIDのCategoryインスタンスをDBから取得する。

        Parameters:
            conn: SQLiteの接続オブジェクト
            category_id: 取得したいカテゴリID

        Returns:
            Categoryインスタンス または None（該当がない場合）
        """
        cur = conn.cursor()
        # カテゴリ基本情報を取得
        cur.execute(f'''
            SELECT id, name, description, type_code, sort_order, created_at, updated_at
            FROM {cls.table_name()} WHERE id=?
        ''', (category_id,))
        row = cur.fetchone()

        if row:
            cat = cls.from_row(row) # 基本情報でオブジェクトを作成
            
            # 中間テーブルから親IDリストを取得
            cur.execute("""
                SELECT parent_category_id
                FROM category_parents
                WHERE child_category_id = ?
            """, (category_id,))
            parent_rows = cur.fetchall()
            cat.parent_ids = [p[0] for p in parent_rows] # 親IDリストを設定

            return cat
        else:
            return None
    
    def to_retriever_category(self) -> retriever_utils.RetrieverCategory:
        """
        Category オブジェクトを RAG 用の RetrieverCategory 型に変換する。
        複数親の場合、現在の RetrieverCategory (特に HierarchicalRetrieverCategory) の設計と整合性を取る必要がある。
        一時的な対応として、階層型で複数親がある場合はエラーとする。
        """
        if self.type_code == "hier":
            if len(self.parent_ids) > 1:
                 # 複数親をHierarchicalRetrieverCategoryにどうマッピングするかは設計次第
                 # 一時的にエラーとするか、最初の親を選ぶか、フラットタグに倒すかなどを決める
                 raise ValueError(f"階層型カテゴリ '{self.name}' (ID: {self.id}) が複数の親を持つため、RetrieverCategoryへの変換に失敗しました。Retriever側の設計を確認してください。")
            # 親IDが0個または1個の場合は、既存のHierarchicalRetrieverCategoryの設計に合わせる
            # parent_id フィールドが str を期待しているので、ID (int) を検索するなどして名前(str)に変換する必要がある
            # 簡単のため、ここでは親ID (int) をそのまま parent_id (str) に渡す (retriever_utils側の修正が必要になる可能性あり)
            # または、親の名前を取得するDBクエリをここに追加する
            parent_name = None
            if self.parent_ids:
                # 親IDに対応する名前を取得するDBクエリを実行する必要がある
                # 例: SELECT name FROM categories WHERE id = self.parent_ids[0]
                # このメソッドがDBコネクションを知らないため、呼び出し元でIDを名前に変換するか、
                # このメソッドに conn を渡すなどの対応が必要になる
                # ここでは簡略化のため、parent_id を None とする (階層構造がRetrieverに伝わらない)
                # TODO: 複数親のRetrieverCategoryへの適切なマッピングを実装する
                pass # 親の名前取得ロジックは別途実装が必要

            # 簡易対応として、階層型だが親情報をRetrieverに渡さないか、単一親の場合のみ渡す
            # RetrieverCategoryのparent_idはOptional[str]なので、親ID(int)から親カテゴリ名(str)への変換が必要
            parent_tagname = None
            if len(self.parent_ids) == 1:
                 # 単一親の場合のみ、その親のカテゴリ名を取得して渡す
                 parent_cat = Category.get_by_id(conn=None, category_id=self.parent_ids[0]) # connが必要になる
                 if parent_cat:
                      parent_tagname = parent_cat.name
                 else:
                      print(f"⚠️ 親カテゴリID {self.parent_ids[0]} が見つかりません。")

            return retriever_utils.HierarchicalRetrieverCategory(
                tagname=self.name,
                parent_id=parent_tagname # 親カテゴリ名 (str) を渡す
            )
        elif self.type_code == "flat":
            return retriever_utils.FlatRetrieverCategory(
                tagname=self.name
            )
        elif self.type_code == "array":
             # ArrayRetrieverCategory が retriever_utils に定義されているか確認
             # もし定義されていればそれを使う
             # 例: return retriever_utils.ArrayRetrieverCategory(tagname=self.name)
             raise NotImplementedError("配列型カテゴリのRetrieverCategory変換は未実装です。")
        else:
            raise ValueError(f"不明な type_code: {self.type_code}")

@dataclass(init=False)
class Paragraph(DBObject):
    id: Optional[int] = None
    document_id: int = None
    parent_id: int = None
    category_id: Optional[int] = None
    order: int = None
    depth: int = None
    name: Optional[str] = None
    body: str = None
    description: Optional[str] = None
    language: Optional[str] = None
    vectorstore_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        document_id: int,
        parent_id: Optional[int],
        category_id: Optional[int],
        order: int,
        depth: int,
        name: Optional[str],
        body: str,
        description: Optional[str] = None,
        language: Optional[str] = None,
        vectorstore_path: Optional[str] = None,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.document_id = document_id
        self.parent_id = parent_id
        self.category_id = category_id
        self.order = order
        self.depth = depth
        self.name = name
        self.body = body
        self.description = description
        self.language = language
        self.vectorstore_path = str(Path(vectorstore_path)) if vectorstore_path else None
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定してください。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "paragraphs"

    @classmethod
    def from_row(cls, row: tuple) -> 'Paragraph':
        return cls(*row)
    
    @classmethod
    def get_languages_by_category_id(cls, conn: sqlite3.Connection, category_id: int) -> list[str]:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT language
            FROM paragraphs
            WHERE category_id = ?
            AND language IS NOT NULL
        """, (category_id,))
        rows = cur.fetchall()
        return [row[0] for row in rows if row[0]]

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO paragraphs (document_id, parent_id, category_id, "order", depth, name, body, description,language,
                                     vectorstore_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ''', (self.document_id, self.parent_id, self.category_id, self.order, self.depth,
              self.name, self.body, self.description, self.language, self.vectorstore_path))
        conn.commit()
        self.id = cur.lastrowid
        return self.id

    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE paragraphs SET document_id=?, parent_id=?, category_id=?, "order"=?, depth=?, name=?, body=?,
            description=?, language=? vectorstore_path=?, updated_at=datetime('now') WHERE id=?
        ''', (self.document_id, self.parent_id, self.category_id, self.order, self.depth,
              self.name, self.body, self.description, self.language, self.vectorstore_path, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM paragraphs WHERE id=?', (self.id,))
        conn.commit()

    def vectorization(
        self,
        conn,
        embedding_name: str = "",
        overwrite: bool = False,
    ):
        if overwrite or not self.vectorstore_path or not os.path.exists(self.vectorstore_path):
            from retriever_utils import FlatRetrieverCategory
            from chain_factory import save_chain_from_text
            cat = Category.get_by_id(conn, self.category_id)
            ret = FlatRetrieverCategory(tagname=cat.name)
            
            success = save_chain_from_text(
                text = self.body,
                vect_path=self.vectorstore_path,
                embedding_name=embedding_name,
                category = ret
            )
            return success

    @classmethod
    def get_vectorstore_by_category_id(cls, conn: sqlite3.Connection, category_id: int) -> list[str]:
        """
        指定されたカテゴリIDに紐づく paragraphs テーブルの vectorstore_path を
        重複なしですべて取得し、リスト形式で返す。
        """
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT vectorstore_path
            FROM paragraphs
            WHERE category_id = ?
            AND vectorstore_path IS NOT NULL
            ORDER BY updated_at DESC
        """, (category_id,))

        rows = cur.fetchall()
        result = [row[0] for row in rows if row[0]]
        
        return result if result else None
    
    @classmethod
    def insert_all(cls, conn: sqlite3.Connection, paragraphs: list["Paragraph"]) -> None:
        print("Paragraph を挿入します")
        for p in paragraphs:
            p.insert(conn)

def create_categories_from_paragraph_tree(
    tree: list[dict],
    dbcon: sqlite3.Connection,
    parent_id: str = None,
    type_code: str = "hier"
) -> list[Category]:
    """
    Paragraph構造に基づきCategoryを自動生成してDBに挿入。
    """
    categories = []

    def _recurse(subtree: list[dict], parent_id: int):
        for order, node in enumerate(subtree):
            cat = Category(
                name=node["name"],
                description=node["body"][:200] if node["body"] else None,
                parent_id=parent_id,
                type_code=type_code,
                sort_order=order,
                dbcon=dbcon,
                insert=True
            )
            categories.append(cat)

            # 子ノードを再帰的に処理
            _recurse(node["children"], parent_id=cat.name)

    _recurse(tree, parent_id=parent_id)
    return categories

@dataclass(init=False)
class Document(DBObject):
    id: Optional[int] = None
    project_id: int = 0
    category_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    file_path: Path = None
    file_type: str = None
    vectorstore_dir: Optional[str] = None
    embedding_model: Optional[str] = None
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    paragraphs: Optional[list[Paragraph]] = field(default_factory=list)

    def __init__(
        self,
        project_id: int,
        category_id: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Path = "",
        vectorstore_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        status: str = "active",
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.project_id = project_id
        self.category_id = category_id
        self.name = name
        self.description = description
        self.file_path = str(Path(file_path))
        self.file_type = document_utils.get_document_type(self.file_path)
        self.vectorstore_dir = str(Path(vectorstore_dir))
        self.embedding_model = embedding_model
        self.status = status
        self.created_at = None
        self.updated_at = None
        
        if not os.path.exists(self.vectorstore_dir):
            print(f"📂 作成対象: {self.vectorstore_dir}")
            os.makedirs(self.vectorstore_dir, exist_ok=True)
        else:
            print(f"📁 既に存在: {self.vectorstore_dir}")

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)
            self.init_paragraphs(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "documents"

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Document':
        return cls(*row)

    def init_paragraphs(self, conn) -> None:
        """
        ツリー構造に従って Paragraph インスタンスを生成し、self.paragraphs に格納する。
        :param tree: build_nested_structure で得られた dict 構造のリスト
        :param category_id: 各パラグラフに共通で割り当てるカテゴリID（任意）
        """
        self.paragraphs = []

        tree = document_utils.build_nested_structure(self.file_path, max_depth=6)

        def _recurse(subtree: list[dict], parent: Optional[Paragraph] = None):
            for node in subtree:
                from langdetect import detect
                body_text = node.get("body", "").strip()

                if not body_text:
                    print(f"⚠️ 空の本文をスキップ: name='{node.get('name', '未命名')}'")
                    _recurse(node.get("children", []), parent=parent)
                    continue  # または continue など文脈に応じて
            
                cat = Category(
                    name=node["name"],
                    description=node["body"][:200] if node["body"] else node["name"],
                    parent_id=parent.category_id if parent else self.category_id,
                    type_code ="hier",
                    dbcon=conn,
                    insert=True
                )

                vectorstore_path = Path(self.vectorstore_dir) / (cat.name+".faiss")
                para = Paragraph(
                    document_id=self.id,
                    parent_id = parent.id if parent else self.category_id,
                    category_id=cat.id,
                    vectorstore_path=str(vectorstore_path),
                    order=node["order"],
                    depth=node["depth"],
                    name=node["name"],
                    body=node["body"],
                    language = detect(body_text),
                    dbcon=conn,
                    insert=True
                )
                para.vectorization(conn, embedding_name=self.embedding_model)
                self.paragraphs.append(para)
                _recurse(node["children"], parent=para)

        _recurse(tree)

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO documents (project_id, category_id, name, description, file_path, file_type, vectorstore_dir,
            embedding_model, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
              self.vectorstore_dir, self.embedding_model, self.status))
        conn.commit()
        self.id = cur.lastrowid
        return self.id

    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE documents SET project_id=?, category_id=?, name=?, description=?, file_path=?, file_type=?,
            vectorstore_dir=?, embedding_model=?, status=?, updated_at=datetime('now')
            WHERE id=?
        ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
              self.vectorstore_dir, self.embedding_model, self.status, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM documents WHERE id=?', (self.id,))
        conn.commit()

@dataclass(init=False)
class Project(DBObject):
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: str = "active"
    default_model_name: str = None
    default_prompt_name: str = None
    default_embedding_name: str = None
    notes: Optional[str] = None
    rag_session: RAGSession = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        author: Optional[str] = None,
        status: str = "active",
        default_model_name: str = None,
        default_prompt_name: str = None,
        default_embedding_name: str = None,
        notes: Optional[str] = None,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.name =name
        self.description = description
        self.author = author
        self.status = status
        self.default_model = default_model_name
        self.default_prompt = default_prompt_name
        self.default_embedding = default_embedding_name
        self.notes = notes
        self.created_at = None
        self.updated_at = None

        if default_model_name and not self.is_model_available(default_model_name): # type: ignore
            raise ValueError(f"指定されたモデル '{default_model_name}' は Ollama に存在しません。ollama run {default_model_name} で導入してください。")

        self.rag_session = RAGSession(
            model_name=default_model_name,
            default_template=default_prompt_name,
            embedding_name=default_embedding_name
        )

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)
    
    @classmethod
    def table_name(cls) -> str:
        return "projects"
    
    def is_model_available(self, model_name) -> bool:
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, check=True
            )
            return model_name in result.stdout
        except Exception as e:
            print(f"⚠️ モデル確認中にエラーが発生しました: {e}")
            return False

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Project':
        return cls(*row)

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO projects (name, description, author, created_at, updated_at, status,
                                default_model, default_prompt, default_embedding, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.name,
            self.description,
            self.author,
            datetime.now().isoformat(),  # または time.strftime("%Y-%m-%d %H:%M:%S")
            datetime.now().isoformat(),
            self.status,
            self.default_model_name,
            self.default_prompt_name,
            self.default_embedding_name,
            self.notes
        ))
        conn.commit()
        self.id = cur.lastrowid
        return self.id


    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE projects SET name=?, description=?, author=?, updated_at=datetime('now'),
            status=?, default_model=?, default_prompt=?, default_embedding=?, notes=? WHERE id=?
        ''', (self.name, self.description, self.author, self.status,
              self.default_model, self.default_prompt, self.default_embedding, self.notes, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM projects WHERE id=?', (self.id,))
        conn.commit()
    
    def start_chat(self, prompt: str) -> str:
        if not self.rag_session:
            raise ValueError("RAGSession is not initialized.")
        
        from ollama import chat  # グローバルollamaがあればそれを使ってもよい

        return chat(
            model=self.rag_session.model_name,
            messages=[{"role": "user", "content": prompt}]
        )["message"]["content"]

# === DB接続ヘルパー ===
def db_connect(db_path: str) -> sqlite3.Connection:
    """SQLite に接続する関数"""
    return sqlite3.connect(db_path)

def db_close(conn: sqlite3.Connection):
    """SQLite 接続を閉じる関数"""
    conn.close()

# === 汎用セレクタ ===
def select_all(conn: sqlite3.Connection, cls: type) -> List[dict]:
    """
    任意のテーブルからデータを辞書形式で全件取得する。
    """
    conn.row_factory = sqlite3.Row  # ← 行を辞書形式で取得可能にする
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {cls.table_name()}")
    rows = cur.fetchall()
    return [dict(row) for row in rows]

# === 汎用関数 ===
def get_category_list(conn: sqlite3.Connection) -> list[dict]:
    """
    カテゴリテーブルからすべてのカテゴリを辞書形式で取得する関数。
    """
    conn.row_factory = sqlite3.Row  # ← 辞書風アクセスを可能にする
    cur = conn.cursor()
    cur.execute("SELECT * FROM categories")
    rows = cur.fetchall()
    return [dict(row) for row in rows]

def get_category_selector(
    conn: sqlite3.Connection,
    parent_id: Optional[int] = None
) -> dict[str, str]:
    """
    指定された親タグに属するカテゴリの {name: description} 辞書を返す。
    parent_id=None のときは全件返す。
    """
    categories = get_category_list(conn)  # ← dictのlistを取得

    result = {}
    for cat in categories:
        if parent_id is None or cat["parent_id"] == parent_id:
            name = cat["name"]
            desc = cat.get("description") or cat["name"]
            result[name] = desc
    return result

# === DB初期化 ===
def init_db(db_path: str, overwrite: bool = False):
    """
    データベースファイルを初期化する。必要に応じて上書き削除可能。

    Parameters:
    - db_path (str | Path): SQLiteデータベースファイルのパス
    - overwrite (bool): True の場合、既存ファイルを削除して再作成
    """
    path = Path(db_path)

    if overwrite and path.exists():
        path.unlink()  # ファイル削除
        print(f"⚠️ 既存のDBファイル {path} を削除しました。")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- プロジェクトテーブル ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            author TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            default_model TEXT,
            default_prompt TEXT,
            default_embedding TEXT ,
            notes TEXT
        );
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS update_project_timestamp
        AFTER UPDATE ON projects
        FOR EACH ROW
        BEGIN
            UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
        END;
    """)

    # --- カテゴリテーブル ---
    # 修正: parent_id カラムを削除
    cur.execute("""
        CREATE TABLE IF NOT EXISTS category_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_code TEXT UNIQUE NOT NULL,
            type_name TEXT NOT NULL,
            description TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            -- parent_id INTEGER,  -- この行を削除
            type_code TEXT NOT NULL DEFAULT 'hier',
            sort_order INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- FOREIGN KEY (parent_id) REFERENCES categories(name) ON DELETE SET DEFAULT ON UPDATE CASCADE, -- この行を削除または修正が必要（元々name参照は非推奨）
            FOREIGN KEY (type_code) REFERENCES category_types(type_code) ON DELETE RESTRICT ON UPDATE CASCADE
        );
    """)

    # 新規追加: カテゴリ間の親子関係を管理する中間テーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS category_parents (
            child_category_id INTEGER NOT NULL,
            parent_category_id INTEGER NOT NULL,
            PRIMARY KEY (child_category_id, parent_category_id), -- 複合主キー
            FOREIGN KEY (child_category_id) REFERENCES categories(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (parent_category_id) REFERENCES categories(id) ON DELETE CASCADE ON UPDATE CASCADE
        );
    """)

    # --- 文書テーブル --- (変更なし)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            category_id INTEGER, -- これは文書が属する単一のカテゴリを指すので変更なし
            name TEXT,
            description TEXT,
            file_path TEXT NOT NULL,
            file_type TEXT DEFAULT 'markdown',
            vectorstore_dir TEXT,
            embedding_model TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL ON UPDATE CASCADE
        );
    """)

    # --- 段落テーブル --- (変更なし)
    # 注意: 段落の parent_id は段落自身の階層構造を指し、カテゴリの親とは異なる
    cur.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            parent_id INTEGER, -- これは段落の親を指すので変更なし
            category_id INTEGER, -- これは段落が属する単一のカテゴリを指すので変更なし
            "order" INTEGER,
            depth INTEGER,
            name TEXT,
            body TEXT,
            description TEXT,
            language TEXT,
            vectorstore_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL ON UPDATE CASCADE
        );
    """)

    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS update_document_timestamp
    AFTER UPDATE ON documents
    FOR EACH ROW
    BEGIN
        UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END;
    """)
    
    # 新規追加: category_parents テーブルの更新トリガーは不要 (複合主キーのため)

    init_tables(db_path)

    conn.commit()
    conn.close()
    print("✅ データベース初期化が完了しました（projects, categories, category_parents, documents, paragraphs）")

def init_tables(db_path="database.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    default_types = [
        ("hier", "階層型", "親子関係によって構成される分類型"),
        ("flat", "フラット型", "階層を持たない独立分類"),
        ("array", "配列型", "複数同時に属するタグ的分類")
    ]
    for code, name, desc in default_types:
        cur.execute("""
            INSERT OR IGNORE INTO category_types (type_code, type_name, description)
            VALUES (?, ?, ?)
        """, (code, name, desc))
    conn.commit()
    conn.close()

# テスト実行
if __name__ == "__main__":
    init_db()
