import sqlite3, os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field

# === 抽象クラス定義 ===
from .database import DBObject  # 相対インポートに変更
from .categories import Category  # 相対インポートに変更

@dataclass(init=False)
class Paragraph(DBObject):
    id: Optional[int] = None
    document_id: int = None
    parent_id: Optional[int] = None # Optional に修正 (最上位段落は親なし)
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
        parent_id: Optional[int], # Optional[int] を受け取る
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
        vectorize: bool = False, # ベクトル化フラグを追加
        embedding_name: Optional[str] = None, # ベクトル化時に使用する埋め込みモデル名
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
        # vectorstore_path は Path オブジェクトの文字列化を保持
        self.vectorstore_path = str(Path(vectorstore_path).resolve()) if vectorstore_path else None
        self.created_at = None
        self.updated_at = None

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定してください。")
            # vectorstore_path を含めて挿入 (NULLABLE に対応)
            self.insert(dbcon)

        # Note: __init__ で挿入した場合、self.id が確定するが、vectorize メソッドは
        # 挿入後にDBからParagraphを取得して呼び出す方が一般的かもしれない。
        # ここでは__init__で続けて呼び出す実装にするが、vectorizationメソッド内で
        # self.id が None の場合はエラーとする必要がある。
        if vectorize:
            if dbcon is None:
                raise ValueError("vectorize=True の場合、dbcon を指定してください。")
            # embedding_name が Optional[str] になったので None チェックを追加
            if not embedding_name:
                 print("⚠️ vectorize=True ですが、embedding_name が指定されていないためベクトル化をスキップします。")
                 # raise ValueError("vectorize=True の場合、embedding_name を指定してください。") # エラーにするか警告にするか
            else:
                # self.id は insert() で確定しているはず
                if self.id is None:
                     print("❌ Paragraph オブジェクトのIDが確定していないためベクトル化できません。")
                     # raise ValueError("Paragraph ID must be set before vectorization.")
                else:
                    self.vectorization(dbcon, embedding_name=embedding_name, overwrite=False) # overwrite はデフォルトFalse


    @classmethod
    def table_name(cls) -> str:
        return "paragraphs"

    @classmethod
    def from_row(cls, row: tuple) -> 'Paragraph':
        # SELECT id, document_id, parent_id, category_id, "order", depth, name, body, description, language, vectorstore_path, created_at, updated_at
        if len(row) != 13:
             raise ValueError(f"Expected 13 columns for Paragraph.from_row, got {len(row)}")

        para = cls.__new__(cls) # __init__ をスキップ
        para.id = row[0]
        para.document_id = row[1]
        para.parent_id = row[2] # NULL の可能性あり
        para.category_id = row[3] # NULL の可能性あり
        para.order = row[4]
        para.depth = row[5]
        para.name = row[6]
        para.body = row[7]
        para.description = row[8]
        para.language = row[9]
        para.vectorstore_path = row[10]
        para.created_at = row[11]
        para.updated_at = row[12]
        return para

    @classmethod
    def get_languages_by_category_id(cls, conn: sqlite3.Connection, category_id: int) -> list[str]:
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT DISTINCT language
                FROM paragraphs
                WHERE category_id = ?
                AND language IS NOT NULL AND language != '' -- 空文字も除外
            """, (category_id,))
            rows = cur.fetchall()
            return [row[0] for row in rows if row[0]]
        except sqlite3.Error as e:
             print(f"Error getting languages for category {category_id}: {e}")
             return []

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        # vectorstore_path は初期挿入時は None の可能性があるため含めない、または nullable に対応
        try:
            cur.execute('''
                INSERT INTO paragraphs (document_id, parent_id, category_id, "order", depth, name, body, description, language,
                                         vectorstore_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''', (self.document_id, self.parent_id, self.category_id, self.order, self.depth,
                  self.name, self.body, self.description, self.language, self.vectorstore_path))
            conn.commit()
            self.id = cur.lastrowid
            # 挿入後に self.id が確定
            return self.id if self.id is not None else -1 # 挿入失敗時は-1などを返す
        except sqlite3.IntegrityError as e:
             # UNIQUE constraint failed: paragraphs.vectorstore_path など
             conn.rollback()
             print(f"Paragraph insertion failed (IntegrityError): {e} - Name: '{self.name}', Path: '{self.vectorstore_path}'")
             raise e # エラーを再発生させる
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Paragraph insertion failed (SQLite Error): {e}")
             raise e # エラーを再発生させる

    def update(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないParagraphオブジェクトは更新できません。")
        cur = conn.cursor()
        try:
            # Note: language=? vectorstore_path=? の間にカンマが抜けていたので追加
            cur.execute('''
                UPDATE paragraphs SET document_id=?, parent_id=?, category_id=?, "order"=?, depth=?, name=?, body=?,
                description=?, language=?, vectorstore_path=?, updated_at=datetime('now') WHERE id=?
            ''', (self.document_id, self.parent_id, self.category_id, self.order, self.depth,
                  self.name, self.body, self.description, self.language, self.vectorstore_path, self.id))
            conn.commit()
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Paragraph update failed (ID: {self.id}): {e}")
             raise e

    def delete(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないParagraphオブジェクトは削除できません。")
        cur = conn.cursor()
        try:
            cur.execute('DELETE FROM paragraphs WHERE id=?', (self.id,))
            conn.commit()
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Paragraph deletion failed (ID: {self.id}): {e}")
             raise e

    def vectorization(
        self,
        conn: sqlite3.Connection, # conn パラメータが必要
        embedding_name: str, # embedding_name は必須にする
        overwrite: bool = False,
    ):
        # self.id が確定しているかチェック
        if self.id is None:
             print("❌ Paragraph オブジェクトのIDが確定していないためベクトル化できません。insert() が正常に完了しているか確認してください。")
             return False

        # ベクトル化対象の本文がない場合はスキップ
        if not self.body or not self.body.strip():
             print(f"⚠️ 段落ID {self.id} ({self.name}) の本文が空です。ベクトル化をスキップします。")
             return False

        # ベクトルストアの保存パスが設定されていない場合はスキップ
        if self.vectorstore_path is None:
             print(f"⚠️ 段落ID {self.id} ({self.name}) の vectorstore_path が設定されていません。ベクトル化をスキップします。")
             return False

        # 既存のベクトルストアの確認 (上書きモードでない場合)
        if not overwrite and os.path.exists(self.vectorstore_path):
            print(f"✅ 既存のベクトルストアが見つかりました: {self.vectorstore_path} (overwrite=False)")
            return True # 既存があれば成功とみなす

        # カテゴリ情報を取得して RetrieverCategory を生成
        # 段落に category_id が設定されていない場合はエラー
        if self.category_id is None:
             print(f"❌ 段落ID {self.id} ({self.name}) に category_id が設定されていません。ベクトル化できません。")
             return False

        cat = Category.get_by_id(conn, self.category_id)
        if not cat:
            print(f"❌ 段落ID {self.id} ({self.name}) に紐づくカテゴリID {self.category_id} が見つかりません。ベクトル化できません。")
            return False

        # Category オブジェクトから RetrieverCategory を生成 (conn を渡す)
        try:
            retriever_cat = cat.to_retriever(conn) # conn を渡す
        except Exception as e:
            # 階層型で複数親の場合などのエラーをここで捕捉
            print(f"❌ カテゴリ '{cat.name}' (ID: {cat.id}) の RetrieverCategory 変換に失敗しました: {e} - ベクトル化をスキップします。")
            return False

        # core.chain_factory は外部ライブラリに依存するため、遅延インポートまたは適切な場所でインポート
        # from core.chain_factory import save_chain_from_text # 相対インポートに変更
        try:
            from core.ingestion import save_text_to_vectorstore
        except ImportError:
            print("❌ core.chain_factory が見つかりません。ベクトル化をスキップします。")
            return False
        except Exception as e:
             print(f"❌ core.chain_factory のインポート中にエラーが発生しました: {e} - ベクトル化をスキップします。")
             return False

        print(f"  ベクトル化実行: 段落ID {self.id} ({self.name}), Path: {self.vectorstore_path}")
        try:
            success = save_text_to_vectorstore(
                text = self.body,
                vect_path=self.vectorstore_path,
                embedding_name=embedding_name,
                category = retriever_cat # 生成した RetrieverCategory を渡す
            )

            # ベクトル化が成功したら、Paragraphオブジェクトのvectorstore_pathが設定されていることを確認し、必要ならDBも更新
            # __init__で既に設定されているはずだが、念のためパスが正しいか確認したり、DBの値を最新にしたりする
            if success:
                # save_chain_from_text は通常、指定された vect_path に保存する
                # DBに保存されている self.vectorstore_path と一致しているか確認
                # self.vectorstore_path は resolve() 済みなので、比較はstr同士でOK
                if self.vectorstore_path != str(Path(self.vectorstore_path).resolve()):
                     # パスが異なる場合はDBを更新する必要があるが、通常は一致するはず
                     # print(f"DEBUG: Paragraph {self.id} path mismatch: DB='{self.vectorstore_path}', Saved='{str(Path(self.vectorstore_path).resolve())}'")
                     pass # 基本的には一致する想定

            return success

        except Exception as e:
             print(f"  ❌ 段落ID {self.id} ({self.name}) のベクトル化中に予期せぬエラーが発生しました: {e}")
             # traceback.print_exc() # 詳細なトレースバックが必要なら有効化
             return False


    @classmethod
    def get_vectorstore_by_category_id(cls, conn: sqlite3.Connection, category_id: int) -> list[str]:
        """
        指定されたカテゴリIDに紐づく paragraphs テーブルの vectorstore_path を
        重複なしですべて取得し、リスト形式で返す。
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT DISTINCT vectorstore_path
                FROM paragraphs
                WHERE category_id = ?
                AND vectorstore_path IS NOT NULL AND vectorstore_path != '' -- NULLや空文字を除外
                ORDER BY updated_at DESC -- 新しいものが優先されるようにソート
            """, (category_id,))

            rows = cur.fetchall()
            # タプル [(path,), (path,), ...] から文字列リスト [path, path, ...] に変換
            result = [row[0] for row in rows if row and row[0]]

            # 結果が空の場合は [] を返す (None ではなく)
            return result if result else []
        except sqlite3.Error as e:
             print(f"Error getting vectorstore paths for category {category_id}: {e}")
             return [] # エラー時も空リストを返す

    @classmethod
    def insert_all(cls, conn: sqlite3.Connection, paragraphs: list["Paragraph"], vectorize: bool = False, embedding_name: Optional[str] = None) -> None:
        print(f"--- {len(paragraphs)} 件の Paragraph を挿入します ---")
        inserted_count = 0
        failed_count = 0
        for i, p in enumerate(paragraphs):
            try:
                # Note: Paragraphの__init__でinsert=Trueにしている場合、ここでinsertを呼ぶと二重挿入になる可能性がある
                # Paragraphオブジェクトの生成と挿入を切り分ける必要がある。
                # 例: init_paragraphs内でinsert=FalseでParagraphオブジェクトリストを作成し、
                # ここでinsert(dbcon=conn, insert=True) のように呼び出す。
                # 現状の__init__ではinsert=Trueがデフォルトなので、ここではinsertを呼ばないように変更する
                # p.insert(conn) # <-- ここを削除または調整

                # または、Paragraph.__init__を insert=False で呼び出す前提にするか、
                # insertメソッド内で self.id が None の場合のみ挿入を実行するように修正が必要。
                # 一旦、ここでは__init__で挿入済みの場合を考慮しないシンプルなループとする（__init__でinsert=True前提）

                # __init__で挿入済みなので、ここではベクトル化のみ実行
                if vectorize and embedding_name:
                     if p.id is None:
                         # __init__でのinsertが失敗した場合
                         print(f"  {i+1}/{len(paragraphs)}: ❌ ID未確定のためベクトル化スキップ - Name:'{p.name}'")
                         failed_count += 1
                         continue # この段落の処理をスキップ

                     success = p.vectorization(conn, embedding_name=embedding_name)
                     if success:
                         print(f"  {i+1}/{len(paragraphs)}: ✅ ID:{p.id}, Name:'{p.name}' - ベクトル化成功")
                         inserted_count += 1 # ベクトル化成功を挿入成功とみなす (ここでは)
                         # ベクトル化パスが更新された可能性があるため、DBを更新 (vectorizationメソッド内でupdateを呼ぶように修正)
                         # p.update(conn) # -> vectorizationメソッド内で適切に更新されるべき
                     else:
                         print(f"  {i+1}/{len(paragraphs)}: ❌ ID:{p.id}, Name:'{p.name}' - ベクトル化失敗")
                         failed_count += 1
                else:
                     # ベクトル化しない場合
                     print(f"  {i+1}/{len(paragraphs)}: ✅ ID:{p.id}, Name:'{p.name}'") # __init__で挿入済み
                     inserted_count += 1


            # __init__でinsert=Trueの場合、例外は__init__で捕捉されるべきだが、念のためここでも捕捉
            except sqlite3.IntegrityError as e:
                 print(f"  {i+1}/{len(paragraphs)}: ❌ 挿入スキップまたはエラー (IntegrityError: {e}) - Name:'{p.name}', Path:'{p.vectorstore_path}'")
                 failed_count += 1
            except Exception as e:
                 print(f"  {i+1}/{len(paragraphs)}: ❌ 処理失敗 (Error: {e}) - Name:'{p.name}'")
                 failed_count += 1

        print(f"--- Paragraph 処理完了: 成功 {inserted_count} 件, 失敗 {failed_count} 件 ---")

# === Paragraph 関連ヘルパー関数 ===
def get_paragraphs_by_document_id(conn: sqlite3.Connection, document_id: int) -> List["Paragraph"]:
    """
    指定された文書IDに紐づくParagraphインスタンスをDBから取得する。

    Parameters:
        conn: SQLiteの接続オブジェクト
        document_id: 取得したい段落が属する文書のID

    Returns:
        Paragraphインスタンスのリスト。エラー時や該当がない場合は空リスト。
    """
    cur = conn.cursor()
    paragraphs = []
    try:
        # Paragraph.from_row が期待するカラム順序でSELECTする
        # SELECT id, document_id, parent_id, category_id, "order", depth, name, body, description, language, vectorstore_path, created_at, updated_at
        table_name = Paragraph.table_name() # "paragraphs"
        cur.execute(f"""
            SELECT id, document_id, parent_id, category_id, "order", depth, name, body, description, language, vectorstore_path, created_at, updated_at
            FROM {table_name}
            WHERE document_id = ?
            ORDER BY "order", id -- 順序とIDでソート
        """, (document_id,))

        rows = cur.fetchall() # [(id, doc_id, parent_id, ...), ...] のタプルリスト
        paragraphs = [Paragraph.from_row(row) for row in rows]

        return paragraphs

    except sqlite3.Error as e:
        print(f"文書ID {document_id} の段落リスト取得中にデータベースエラーが発生しました: {e}")
        return [] # エラー時は空リストを返す
    except ValueError as e:
         # from_row が失敗した場合など
         print(f"文書ID {document_id} の段落データ形式エラー: {e}")
         return []
    except Exception as e:
        # その他の予期せぬエラー
        print(f"文書ID {document_id} の段落リスト取得中に予期せぬエラーが発生しました: {e}")
        return []
