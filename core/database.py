import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Any, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import document_utils
import retriever_utils

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
class Project(DBObject):
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: str = "active"
    default_prompt: str = "japanese_concise"
    default_embedding: str = "nomic-embed-text:latest"
    notes: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        author: Optional[str] = None,
        status: str = "active",
        default_prompt: str = "japanese_concise",
        default_embedding: str = "nomic-embed-text:latest",
        notes: Optional[str] = None,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.name =name
        self.description = description
        self.author = author
        self.status = status
        self.default_prompt = default_prompt
        self.default_embedding = default_embedding
        self.notes = notes
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "projects"

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Project':
        return cls(*row)

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO projects (name, description, author, created_at, updated_at, status, default_prompt, default_embedding, notes)
            VALUES (?, ?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?)
        ''', (self.name, self.description, self.author, self.status, self.default_prompt, self.default_embedding, self.notes))
        conn.commit()
        self.id = cur.lastrowid
        return self.id

    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE projects SET name=?, description=?, author=?, updated_at=datetime('now'),
            status=?, default_prompt=?, default_embedding=?, notes=? WHERE id=?
        ''', (self.name, self.description, self.author, self.status,
              self.default_prompt, self.default_embedding, self.notes, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM projects WHERE id=?', (self.id,))
        conn.commit()

@dataclass(init=False)
class Category(DBObject):
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    parent_tag: str = "root"
    type_code: str = "hier"
    sort_order: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        parent_tag: str = "root",
        type_code: str = "hier",
        sort_order: int = 0,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.name = name
        self.description = description
        self.parent_tag = parent_tag
        self.type_code = type_code
        self.sort_order = sort_order
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "categories"
    
    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Category':
        return cls(*row)

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO categories (name, description, parent_tag, type_code, sort_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ''', (self.name, self.description, self.parent_tag, self.type_code, self.sort_order))
        conn.commit()
        self.id = cur.lastrowid
        return self.id

    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE categories SET name=?, description=?, parent_tag=?, type_code=?, sort_order=?, updated_at=datetime('now')
            WHERE id=?
        ''', (self.name, self.description, self.parent_tag, self.type_code, self.sort_order, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM categories WHERE id=?', (self.id,))
        conn.commit()
    
    def to_retriever_category(self) -> retriever_utils.RetrieverCategory:
        """
        Category オブジェクトを RAG 用の RetrieverCategory 型に変換する。
        """
        if self.type_code == "hier":
            return retriever_utils.HierarchicalRetrieverCategory(
                tagname=self.name,
                parent_tag=self.parent_tag or "root"
            )
        elif self.type_code == "flat":
            return retriever_utils.FlatRetrieverCategory(
                tagname=self.name
            )
        else:
            raise ValueError(f"不明な type_code: {self.type_code}")

@dataclass(init=False)
class Document(DBObject):
    id: Optional[int] = None
    project_id: int = 0
    category_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    file_path: Path = None
    file_type: str = None
    vectorstore_path: Optional[str] = None
    embedding_model: Optional[str] = None
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        project_id: int,
        category_id: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Path = "",
        vectorstore_path: Optional[Path] = None,
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
        self.vectorstore_path = vectorstore_path
        self.embedding_model = embedding_model
        self.status = status
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "documents"

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Document':
        return cls(*row)

    def _init_paragraph(
            self,
            name: Optional[str] = None,
            description: Optional[str] = None,
            file_path: Optional[Path] = None,
            dbcon: Optional[sqlite3.Connection] = None,
            insert: bool = True
        ) -> 'Paragraph':
            if self.id is None:
                raise ValueError("Paragraph を作成する前に Document を DB に保存してください（id が必要です）")

            para = Paragraph(
                document_id=self.id,
                project_id=self.project_id,
                category_id=self.category_id,
                name=name or self.name,
                description=description or self.description,
                file_path=file_path or Path(self.file_path),
                vectorstore_path=self.vectorstore_path,
                embedding_model=self.embedding_model,
                status=self.status,
                dbcon=dbcon,
                insert=insert
            )

            self.paragraphs.append(para)
            return para

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO documents (project_id, category_id, name, description, file_path, file_type, vectorstore_path,
            embedding_model, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
              self.vectorstore_path, self.embedding_model, self.status))
        conn.commit()
        self.id = cur.lastrowid
        return self.id

    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE documents SET project_id=?, category_id=?, name=?, description=?, file_path=?, file_type=?,
            vectorstore_path=?, embedding_model=?, status=?, updated_at=datetime('now')
            WHERE id=?
        ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
              self.vectorstore_path, self.embedding_model, self.status, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM documents WHERE id=?', (self.id,))
        conn.commit()

@dataclass(init=False)
class Paragraph(DBObject):
    id: Optional[int] = None
    project_id: int = 0
    category_id: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    file_path: Path = None
    file_type: str = None
    vectorstore_path: Optional[str] = None
    embedding_model: Optional[str] = None
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        project_id: int,
        category_id: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Path = "",
        vectorstore_path: Optional[Path] = None,
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
        self.vectorstore_path = vectorstore_path
        self.embedding_model = embedding_model
        self.status = status
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "paragraphs"

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Paragraph':
        return cls(*row)

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO paragraphs (project_id, category_id, name, description, file_path, file_type, vectorstore_path,
            embedding_model, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
              self.vectorstore_path, self.embedding_model, self.status))
        conn.commit()
        self.id = cur.lastrowid
        return self.id

    def update(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
            UPDATE paragraphs SET project_id=?, category_id=?, name=?, description=?, file_path=?, file_type=?,
            vectorstore_path=?, embedding_model=?, status=?, updated_at=datetime('now')
            WHERE id=?
        ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
              self.vectorstore_path, self.embedding_model, self.status, self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('DELETE FROM paragraphs WHERE id=?', (self.id,))
        conn.commit()

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
    parent_tag: Optional[str] = None
) -> dict[str, str]:
    """
    指定された親タグに属するカテゴリの {name: description} 辞書を返す。
    parent_tag=None のときは全件返す。
    """
    categories = get_category_list(conn)  # ← dictのlistを取得

    result = {}
    for cat in categories:
        if parent_tag is None or cat["parent_tag"] == parent_tag:
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
        default_prompt TEXT DEFAULT 'japanese_concise',
        default_embedding TEXT DEFAULT 'nomic-embed-text:latest',
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
        parent_tag TEXT DEFAULT 'root',
        type_code TEXT NOT NULL DEFAULT 'hier',
        sort_order INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (parent_tag) REFERENCES categories(name) ON DELETE SET DEFAULT ON UPDATE CASCADE,
        FOREIGN KEY (type_code) REFERENCES category_types(type_code) ON DELETE RESTRICT ON UPDATE CASCADE
    );
    """)

    # --- 文書テーブル ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        category_id INTEGER,
        name TEXT,
        description TEXT,
        file_path TEXT NOT NULL,
        file_type TEXT DEFAULT 'markdown',
        vectorstore_path TEXT,
        embedding_model TEXT,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE ON UPDATE CASCADE,
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

    init_tables(db_path)

    conn.commit()
    conn.close()
    print("✅ データベース初期化が完了しました（projects, categories, documents）")

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
