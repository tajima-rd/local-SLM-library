# --- START OF FILE database.py ---

import sqlite3, os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
import subprocess # Added for ollama list check

# === 抽象クラス定義 ===

from abc import ABC, abstractmethod

# === データモデル定義 ===
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

def init_db(db_path: str = "database.db", overwrite: bool = False):
    """
    データベースファイルを初期化する。必要に応じて上書き削除可能。

    Parameters:
    - db_path (str): SQLiteデータベースファイルのパス
    - overwrite (bool): True の場合、既存ファイルを削除して再作成
    """
    path = Path(db_path)

    if overwrite and path.exists():
        try:
            path.unlink()  # ファイル削除
            print(f"⚠️ 既存のDBファイル {path} を削除しました。")
        except OSError as e:
            print(f"❌ 既存のDBファイル {path} の削除に失敗しました: {e}")
            # 削除に失敗しても続行するかどうかは要件による
            pass

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # --- プロジェクトテーブル ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE, -- プロジェクト名はユニークが良いだろう
                description TEXT,
                author TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                default_model TEXT,
                default_prompt TEXT,
                default_embedding TEXT,
                notes TEXT
            );
        """)

        # トリガーは updated_at フィールドを TIMESTAMP DEFAULT CURRENT_TIMESTAMP にすれば不要（INSERT時）
        # UPDATE 時のトリガーは有効
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS update_project_timestamp
            AFTER UPDATE ON projects
            FOR EACH ROW
            BEGIN
                UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
        """)

        # --- カテゴリタイプテーブル ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS category_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type_code TEXT UNIQUE NOT NULL,
                type_name TEXT NOT NULL,
                description TEXT
            );
        """)

        # --- カテゴリテーブル ---
        # level カラムを追加 (マイグレーション処理は削除)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                type_code TEXT NOT NULL DEFAULT 'hier',
                sort_order INTEGER DEFAULT 0,
                level INTEGER DEFAULT 0, -- 新規追加: 階層レベル (ルートは0)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (type_code) REFERENCES category_types(type_code) ON DELETE RESTRICT ON UPDATE CASCADE
            );
        """)

        # カテゴリ間の親子関係を管理する中間テーブル
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
                file_path TEXT NOT NULL UNIQUE, -- ファイルパスはユニークが良いだろう
                file_type TEXT DEFAULT 'markdown',
                vectorstore_dir TEXT,
                embedding_model TEXT, -- embedding_model カラムを追加
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
                parent_id INTEGER, -- これは段落の親を指すので変更なし (NULL 許容)
                category_id INTEGER, -- これは段落が属する単一のカテゴリを指すので変更なし (NULL 許容)
                "order" INTEGER,
                depth INTEGER,
                name TEXT,
                body TEXT,
                description TEXT,
                language TEXT,
                vectorstore_path TEXT UNIQUE, -- ベクトルストアパスはユニークが良いだろう
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

        # paragraphs テーブルの更新トリガーも追加
        cur.execute("""
        CREATE TRIGGER IF NOT EXISTS update_paragraph_timestamp
        AFTER UPDATE ON paragraphs
        FOR EACH ROW
        BEGIN
            UPDATE paragraphs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
        END;
        """)

        # category_parents テーブルの更新トリガーは不要 (複合主キーのため)
        # categories テーブルの更新トリガーも updated_at のみでlevelは自動更新されない点に注意

        # デフォルトのカテゴリタイプを挿入
        init_tables(db_path)

        conn.commit()
        print("✅ データベース初期化が完了しました（projects, categories, category_parents, documents, paragraphs, category_types）")

    except sqlite3.Error as e:
        print(f"❌ データベース初期化中にエラーが発生しました: {e}")
        if conn:
            conn.rollback() # エラー発生時はロールバック
    finally:
        if conn:
            conn.close()

def init_tables(db_path="database.db"):
    """デフォルトのカテゴリタイプを category_types テーブルに挿入する"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        default_types = [
            ("hier", "階層型", "親子関係によって構成される分類型"),
            ("flat", "フラット型", "階層を持たない独立分類"),
            ("array", "配列型", "複数同時に属するタグ的分類") # 将来の拡張用
        ]
        for code, name, desc in default_types:
            cur.execute("""
                INSERT OR IGNORE INTO category_types (type_code, type_name, description)
                VALUES (?, ?, ?)
            """, (code, name, desc))
        conn.commit()
        # print("✅ デフォルトカテゴリタイプを挿入しました。") # 毎回出すと冗長なのでコメントアウト
    except sqlite3.Error as e:
        print(f"❌ デフォルトカテゴリタイプの挿入中にエラーが発生しました: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def select_all(conn: sqlite3.Connection, cls: type) -> List[dict]:
    """
    任意のテーブルからデータを辞書形式で全件取得する。
    """
    # 一時的に row_factory を設定
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # level カラムが追加された categories テーブルから取得する場合、levelも含まれるようになる
        cur.execute(f"SELECT * FROM {cls.table_name()}")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Error selecting from {cls.table_name()}: {e}")
        return []
    finally:
        # 処理が終わったら元の row_factory に戻す
        conn.row_factory = original_row_factory

def db_connect(db_path: str) -> sqlite3.Connection:
    """SQLite に接続する関数"""
    # check_same_thread=False はマルチスレッド環境で必要になることがあるが、基本は True が安全
    # UI などで利用する場合は検討
    # isolation_level=None はオートコミットモードにする。明示的に conn.commit() する場合は None にしない
    # デフォルトは 'DEFERRED' なので、ここではNoneにせず明示的なコミットを使う
    try:
         return sqlite3.connect(db_path)
    except sqlite3.Error as e:
         print(f"データベース '{db_path}' への接続に失敗しました: {e}")
         raise # 接続失敗は致命的なので例外を発生させる

def db_close(conn: sqlite3.Connection):
    """SQLite 接続を閉じる関数"""
    conn.close()

# --- END OF FILE database.py ---