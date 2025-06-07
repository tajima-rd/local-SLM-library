import sqlite3, os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
import subprocess  # Added for ollama list check

# === 抽象クラス定義 ===
from core import document_utils  # 相対インポートに変更
from core import rag_session as rag  # 相対インポートに変更
from .database import DBObject  # 相対インポートに変更
from .database import db_connect, db_close  # 相対インポートに変更

from .paragraphs import Paragraph  # type: ignore # 相対インポートに変更

@dataclass(init=False)
class Project(DBObject):
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: str = "active"
    default_model: Optional[str] = None # default_model_name から変更
    default_prompt: Optional[str] = None # default_prompt_name から変更
    default_embedding: Optional[str] = None # default_embedding_name から変更
    notes: Optional[str] = None
    rag_session: Optional[rag.RAGSession] = None # Optional に変更

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        author: Optional[str] = None,
        status: str = "active",
        default_model_name: Optional[str] = None, # __init__ パラメータ名は維持
        default_prompt_name: Optional[str] = None, # __init__ パラメータ名は維持
        default_embedding_name: Optional[str] = None, # __init__ パラメータ名は維持
        notes: Optional[str] = None,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.name =name
        self.description = description
        self.author = author
        self.status = status
        self.default_model = default_model_name # フィールド名に代入
        self.default_prompt = default_prompt_name # フィールド名に代入
        self.default_embedding = default_embedding_name # フィールド名に代入
        self.notes = notes
        self.created_at = None
        self.updated_at = None

        # モデル存在チェックはデフォルトモデルが設定されている場合のみ行う
        if default_model_name and not self.is_model_available(default_model_name):
            # 警告ではなくエラーとして処理を中断
            raise ValueError(f"指定されたモデル '{default_model_name}' は Ollama に存在しません。ollama run {default_model_name} で導入してください。")

        # RAGSession の初期化は、必要な情報が揃っていて、かつセッションが必要な場合に遅延させるか、
        # ここで初期化するなら全てのパラメータを Optional に対応させる
        # セッションオブジェクトはDBに保存しないため、DBからロードしたProjectオブジェクトには含まれない
        # ここでは __init__ で渡されたパラメータで初期化しておく
        # RAGSession は初期化時にモデル存在チェックを行うため、ここでのチェックと重複するが、
        # Project オブジェクト作成時にモデルの有効性を確認できるメリットがある。
        # RAGSession 初期化はここでは行わない方が、DBロード時の from_row で
        # 不要な RAGSession オブジェクト生成を防げる。
        # self.rag_session は None のままにしておき、get_rag_session で遅延初期化する。
        self.rag_session = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "projects"

    def is_model_available(self, model_name) -> bool:
        """指定されたモデル名が Ollama に存在するかチェックする"""
        # subprocess はファイル冒頭でインポート済み
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, check=True, timeout=10 # タイムアウト設定を追加
            )
            # ollama list の出力は "model_name:version tag\n...". tag の前にスペースがある
            # モデル名が完全に一致するか、tag名として含まれているかをチェック
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1: # ヘッダー行以外をチェック
                 # 各行の最初のスペースまでがモデル名とタグなので、それを比較
                 for line in lines[1:]:
                     # 行の先頭から最初の空白または改行までを取得し、一致するか確認
                     parts = line.split()
                     if parts and parts[0] == model_name:
                         return True
            return False # ヘッダー行しかなかった場合、または一致するモデルが見つからなかった場合

        except FileNotFoundError:
            print(f"⚠️ 'ollama' コマンドが見つかりません。Ollamaがインストールされ、パスが通っているか確認してください。")
            return False
        except subprocess.CalledProcessError as e:
            print(f"⚠️ ollama list 実行中にエラーが発生しました (終了コード: {e.returncode}): {e.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
             print(f"⚠️ ollama list 実行がタイムアウトしました。")
             return False
        except Exception as e:
            print(f"⚠️ モデル確認中に予期せぬエラーが発生しました: {e}")
            return False

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Project':
        # SELECT id, name, description, author, created_at, updated_at, status, default_model, default_prompt, default_embedding, notes
        if len(row) != 11:
             raise ValueError(f"Expected 11 columns for Project.from_row, got {len(row)}")

        proj = cls.__new__(cls) # __init__ をスキップ
        proj.id = row[0]
        proj.name = row[1]
        proj.description = row[2]
        proj.author = row[3]
        proj.created_at = row[4]
        proj.updated_at = row[5]
        proj.status = row[6]
        proj.default_model = row[7]
        proj.default_prompt = row[8]
        proj.default_embedding = row[9]
        proj.notes = row[10]
        proj.rag_session = None # DBからロードした時点では RAGSession は生成しない
        return proj

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        try:
            cur.execute('''
                INSERT INTO projects (name, description, author, created_at, updated_at, status,
                                    default_model, default_prompt, default_embedding, notes)
                VALUES (?, ?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?)
            ''', (
                self.name,
                self.description,
                self.author,
                self.status,
                self.default_model, # フィールド名を使用
                self.default_prompt, # フィールド名を使用
                self.default_embedding, # フィールド名を使用
                self.notes
            ))
            conn.commit()
            self.id = cur.lastrowid
            print(f"✅ Project を挿入しました: ID={self.id}, Name='{self.name}'")
            return self.id if self.id is not None else -1 # 挿入失敗時は-1などを返す
        except sqlite3.IntegrityError as e:
             conn.rollback()
             print(f"Project insertion failed (IntegrityError): {e} - Name: '{self.name}'")
             raise e # エラーを再発生させる
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Project insertion failed (SQLite Error): {e}")
             raise e # エラーを再発生させる


    def update(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないProjectオブジェクトは更新できません。")
        cur = conn.cursor()
        try:
            cur.execute('''
                UPDATE projects SET name=?, description=?, author=?, updated_at=datetime('now'),
                status=?, default_model=?, default_prompt=?, default_embedding=?, notes=? WHERE id=?
            ''', (self.name, self.description, self.author, self.status,
                  self.default_model, self.default_prompt, self.default_embedding, self.notes, self.id)) # フィールド名を使用
            conn.commit()
            print(f"✅ Project を更新しました: ID={self.id}, Name='{self.name}'")
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Project update failed (ID: {self.id}): {e}")
             raise e


    def delete(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないProjectオブジェクトは削除できません。")
        cur = conn.cursor()
        try:
            cur.execute('DELETE FROM projects WHERE id=?', (self.id,))
            conn.commit()
            print(f"✅ Project を削除しました: ID={self.id}, Name='{self.name}'")
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Project deletion failed (ID: {self.id}): {e}")
             raise e

    def get_rag_session(self) -> rag.RAGSession:
         """RAGSession オブジェクトを取得/初期化する (遅延初期化)"""
         if self.rag_session is None:
              # デフォルト設定を使って RAGSession を初期化
              # ollama が利用可能かなどのチェックは RAGSession 内部で行われると良い
              # RAGSession コンストラクタは Optional を受け取る
              try:
                # embedding_name が設定されていない場合は RAGSession に渡さない
                rag_embedding_name = self.default_embedding if self.default_embedding else None
                self.rag_session = RAGSession(
                    model_name=self.default_model,
                    default_template=self.default_prompt,
                    embedding_name=rag_embedding_name # None の可能性あり
                )
                print(f"✅ Project ID {self.id} の RAGSession を初期化しました (Model: {self.default_model}, Embedding: {self.default_embedding})。")
              except Exception as e:
                   # RAGSession 初期化失敗は致命的である可能性が高い
                   print(f"❌ Project ID {self.id} の RAGSession 初期化に失敗しました: {e}")
                   # 初期化失敗時は None のままにするか、エラーを再発生させるか。
                   # ここではエラーを再発生させて、呼び出し元で捕捉させる。
                   raise ValueError(f"RAGSession の初期化に失敗しました。プロジェクトのデフォルト設定を確認してください: {e}") from e
         return self.rag_session

    # TODO: RAGSession を使用した chat/invoke メソッドを Project に追加する
    # 現在の start_chat はシンプルすぎるため、実際のRAGセッションと連携させる必要がある
    # def start_chat(self, prompt: str) -> str:
    #     session = self.get_rag_session()
    #     # TODO: ここで Retriever を取得し、Sessionを使ってチャットを実行するロジック
    #     # 例えば、session.invoke(prompt, retriever=...) のようなメソッドが必要になる
    #     print("Warning: Project.start_chat is not fully implemented for RAG.")
    #     # とりあえず Ollama chat を直接呼び出す
    #     from ollama import chat # 遅延インポート
    #     messages = [{"role": "user", "content": prompt}]
    #     try:
    #         response = chat(
    #            model=session.model_name, # セッションからモデル名を取得
    #            messages=messages
    #         )
    #         return response["message"]["content"]
    #     except Exception as e:
    #         print(f"⚠️ チャット中にエラーが発生しました: {e}")
    #         return f"エラーが発生しました: {e}"

# === Project 関連ヘルパー関数 ===
def get_project_names(conn: sqlite3.Connection) -> List[str]:
    """
    DBからすべてのプロジェクト名を取得し、リスト形式で返す。

    Parameters:
        conn: SQLiteの接続オブジェクト

    Returns:
        プロジェクト名の文字列リスト。エラー時は空リスト。
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM projects ORDER BY name") # 名前順でソート
        rows = cur.fetchall()
        # [(name,), (name,), ...] のタプルリストから文字列リストに変換
        return [row[0] for row in rows if row and row[0] is not None]
    except sqlite3.Error as e:
        print(f"プロジェクト名の一覧取得中にエラーが発生しました: {e}")
        return [] # エラー時は空リストを返す

def load_project_by_name(db_path: str, project_name: str) -> Optional[Project]:
    """
    指定された名前の Project インスタンスをDBから取得する。

    Parameters:
        db_path: SQLiteデータベースファイルのパス
        project_name: 取得したいプロジェクトの名前

    Returns:
        Projectインスタンス または None（該当がない場合）
    """
    conn = None
    try:
        conn = db_connect(db_path)
        cur = conn.cursor()

        # from_row が期待するカラム順序でSELECTする
        cur.execute("""
            SELECT id, name, description, author, created_at, updated_at, status, default_model, default_prompt, default_embedding, notes
            FROM projects
            WHERE name = ?
        """, (project_name,))

        row = cur.fetchone()

        if row:
            # Rowが見つかったら Project.from_row を使ってインスタンスを生成
            project = Project.from_row(row)
            # from_row は __init__ をスキップするため、rag_session は None のまま。
            # これは get_rag_session() で遅延初期化される。
            return project
        else:
            # 該当するプロジェクトが見つからなかった場合
            print(f"プロジェクト '{project_name}' が見つかりませんでした。")
            return None

    except sqlite3.Error as e:
        print(f"データベースエラーが発生しました (プロジェクト '{project_name}' ロード時): {e}")
        return None # データベースエラーの場合は None を返すか、例外を再発生させる

    except ValueError as e:
         # from_row が失敗した場合など
         print(f"プロジェクトデータ形式エラー (プロジェクト '{project_name}' ロード時): {e}")
         return None
    except Exception as e:
        # その他の予期せぬエラー
        print(f"予期せぬエラーが発生しました (プロジェクト '{project_name}' ロード時): {e}")
        return None # あるいは例外を再発生させる
    finally:
        if conn:
            db_close(conn)