import sqlite3
from pathlib import Path
from typing import Optional, List, Any, Tuple
from dataclasses import dataclass, asdict, field

# === 抽象クラス定義 ===
from core import document_utils  # 相対インポートに変更
from .database import DBObject  # 相対インポートに変更
from .paragraphs import Paragraph  # type: ignore # 相対インポートに変更

@dataclass(init=False)
class Document(DBObject):
    id: Optional[int] = None
    project_id: int = 0
    category_id: Optional[int] = None # 文書全体が属するカテゴリID (任意)
    name: Optional[str] = None
    description: Optional[str] = None
    file_path: str = "" # Path オブジェクトではなく str で保持
    file_type: str = None
    vectorstore_dir: Optional[str] = None # Path オブジェクトではなく str で保持
    embedding_model: Optional[str] = None # embedding_model フィールドを追加
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    paragraphs: List[Paragraph] = field(default_factory=list) # Paragraph リストを格納 (DBには保存されない)

    def __init__(
        self,
        project_id: int,
        category_id: Optional[int] = None, # Document 自体が属するカテゴリ
        name: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Path = Path(""), # Path オブジェクトで受け取る
        vectorstore_dir: Optional[Path] = None, # Path オブジェクトで受け取る
        embedding_model: Optional[str] = None, # embedding_model を __init__ に追加
        status: str = "active",
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
        paragraphs: Optional[List[Paragraph]] = None, # 初期化時に段落を指定できるようにする
    ):
        self.id = None
        self.project_id = project_id
        self.category_id = category_id 
        self.name = name or Path(file_path).name 
        self.description = description
        self.file_path = str(file_path.resolve()) if file_path else ""
        self.file_type = document_utils.get_document_type(self.file_path) if self.file_path else "unknown"
        self.vectorstore_dir = str(vectorstore_dir.resolve()) if vectorstore_dir else None
        self.embedding_model = embedding_model # フィールドに設定
        self.status = status
        self.created_at = None
        self.updated_at = None
        self.paragraphs = paragraphs if paragraphs is not None else [] # 初期化時に段落リストを設定

        # vectorstore_dir の作成は self.id がなくても可能なので __init__ で実行
        if self.vectorstore_dir:
            try:
                path_obj = Path(self.vectorstore_dir)
                if not path_obj.exists():
                    print(f"📂 vectorstore_dir を作成: {self.vectorstore_dir}")
                    path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                 print(f"❌ vectorstore_dir '{self.vectorstore_dir}' の作成に失敗しました: {e}")


        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            self.insert(dbcon)

    @classmethod
    def table_name(cls) -> str:
        return "documents"

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Document':
        # SELECT id, project_id, category_id, name, description, file_path, file_type, vectorstore_dir, embedding_model, status, created_at, updated_at
        if len(row) != 12:
             raise ValueError(f"Expected 12 columns for Document.from_row, got {len(row)}")

        doc = cls.__new__(cls) # __init__ をスキップ
        doc.id = row[0]
        doc.project_id = row[1]
        doc.category_id = row[2] # NULL の可能性あり
        doc.name = row[3]
        doc.description = row[4]
        doc.file_path = row[5]
        doc.file_type = row[6]
        doc.vectorstore_dir = row[7] # NULL の可能性あり
        doc.embedding_model = row[8] # NULL の可能性あり
        doc.status = row[9]
        doc.created_at = row[10]
        doc.updated_at = row[11]
        doc.paragraphs = [] # from_row 時点では段落はロードしない。別途 load_paragraphs メソッドが必要
        return doc

    def load_paragraphs(self, conn: sqlite3.Connection) -> None:
         """DBからこのDocumentに紐づくParagraphsをロードし、self.paragraphsに設定する。"""
         if self.id is None:
              print("警告: 文書IDがありません。段落をロードできません。")
              self.paragraphs = []
              return

         cur = conn.cursor()
         try:
              cur.execute("""
                  SELECT id, document_id, parent_id, category_id, "order", depth, name, body, description, language, vectorstore_path, created_at, updated_at
                  FROM paragraphs
                  WHERE document_id = ?
                  ORDER BY "order", id -- 順序とIDでソート
              """, (self.id,))
              rows = cur.fetchall()
              self.paragraphs = [Paragraph.from_row(row) for row in rows]
              # print(f"✅ 文書 ID {self.id} に紐づく {len(self.paragraphs)} 件の段落をロードしました。") # ロード成功メッセージは控えめに
         except sqlite3.Error as e:
              print(f"Error loading paragraphs for document {self.id}: {e}")
              self.paragraphs = []

    def insert(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        try:
            cur.execute('''
                INSERT INTO documents (project_id, category_id, name, description, file_path, file_type, vectorstore_dir,
                embedding_model, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
                  self.vectorstore_dir, self.embedding_model, self.status))
            conn.commit()
            self.id = cur.lastrowid
            print(f"✅ Document を挿入しました: ID={self.id}, Name='{self.name}', Path='{self.file_path}'")
            return self.id if self.id is not None else -1 # 挿入失敗時は-1などを返す
        except sqlite3.IntegrityError as e:
             conn.rollback()
             print(f"Document insertion failed (IntegrityError): {e} - Name: '{self.name}', Path: '{self.file_path}'")
             raise e # エラーを再発生させる
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Document insertion failed (SQLite Error): {e}")
             raise e # エラーを再発生させる

    def update(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないDocumentオブジェクトは更新できません。")
        cur = conn.cursor()
        try:
            cur.execute('''
                UPDATE documents SET project_id=?, category_id=?, name=?, description=?, file_path=?, file_type=?,
                vectorstore_dir=?, embedding_model=?, status=?, updated_at=datetime('now')
                WHERE id=?
            ''', (self.project_id, self.category_id, self.name, self.description, self.file_path, self.file_type,
                  self.vectorstore_dir, self.embedding_model, self.status, self.id))
            conn.commit()
            print(f"✅ Document を更新しました: ID={self.id}, Name='{self.name}'")
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Document update failed (ID: {self.id}): {e}")
             raise e

    def delete(self, conn: sqlite3.Connection):
        if self.id is None:
            raise ValueError("IDがないDocumentオブジェクトは削除できません。")
        cur = conn.cursor()
        try:
            # documents テーブルから削除 -> paragraphs テーブルの対応するエントリは CASCADE で削除される
            cur.execute('DELETE FROM documents WHERE id=?', (self.id,))
            conn.commit()
            print(f"✅ Document を削除しました: ID={self.id}, Name='{self.name}'")
        except sqlite3.Error as e:
             conn.rollback()
             print(f"Document deletion failed (ID: {self.id}): {e}")
             raise e

# === Document 関連ヘルパー関数 ===
def init_paragraphs(
    conn: sqlite3.Connection,
    document: Document,
    category_id: int, # 各段落に紐づけるカテゴリID
    vectorize: bool = False,
    embedding_name: Optional[str] = None
):
    """
    Document オブジェクトのファイルパスから段落を抽出し、
    Paragraph オブジェクトを作成、DBに挿入し、必要に応じてベクトル化する。

    Parameters:
        conn: SQLiteの接続オブジェクト
        document: 段落を作成する対象のDocumentオブジェクト (IDが確定している必要がある)
        category_id: 作成される各段落に紐づけるカテゴリID
        vectorize: 段落挿入後にベクトル化を実行するかどうか
        embedding_name: ベクトル化に使用するEmbeddingモデル名 (vectorize=True の場合必須)
    """
    if document.id is None:
        raise ValueError("Document オブジェクトがDBに挿入されておらず、IDが確定していません。")

    if not document.file_path or not Path(document.file_path).exists():
         print(f"❌ 文書ファイルが見つかりません: {document.file_path}")
         return # 処理を中断

    if vectorize and not embedding_name:
         raise ValueError("vectorize=True の場合、embedding_name を指定する必要があります。")

    print(f"\n--- 文書 '{document.name}' (ID: {document.id}) から段落を初期化 ---")

    try:
        # document_utils から段落構造を抽出
        # extract_paragraphs は (depth, name, body, description) のリストを返す想定
        structured_paragraphs = document_utils.extract_paragraphs(document.file_path)
        
        # Paragraph オブジェクトリストを作成
        paragraph_objects = []
        order_counter = 0
        parent_id_map = {0: None} # depth 0 の親は None

        # ベクトルストア保存ディレクトリが指定されていない場合はエラーとするか、デフォルトパスを生成する
        if document.vectorstore_dir is None:
             # デフォルトパス生成の例: ./vectorstores/project_[id]/doc_[id]/
             default_vect_dir = Path("./vectorstores") / f"project_{document.project_id}" / f"doc_{document.id}"
             print(f"ℹ️ Document に vectorstore_dir が指定されていません。デフォルトパス '{default_vect_dir}' を使用します。")
             document.vectorstore_dir = str(default_vect_dir.resolve())
             # DBのdocumentレコードを更新する必要があるが、ここでは簡略化のため省略
             # 必要に応じて document.update(conn) を呼び出す


        for item in structured_paragraphs:
            depth, name, body, description = item

            # 各段落のベクトルストアパスを生成
            # 例: [vectorstore_dir]/para_[paragraph_id].vect (挿入前はIDがないので一意なファイル名にする必要がある)
            # Document ID + order + name hash などで一意なファイル名を生成するか、
            # 挿入してID確定後にパスを設定・更新するか。
            # ここでは、挿入後にIDを元にパスを生成し、update を行う方式とする。
            # 初期挿入時は vectorstore_path = NULL としておく。
            
            # 段落オブジェクトを作成 (insert=Falseで、後でまとめてinsert_allする前提)
            # __init__ で insert=True がデフォルトなので、insert=False を明示的に指定する
            # vectorization は insert_all でまとめて行う
            para_obj = Paragraph(
                document_id=document.id,
                parent_id=parent_id_map.get(depth - 1, None), # 親の depth を取得
                category_id=category_id, # 全段落に指定されたカテゴリを紐づける
                order=order_counter,
                depth=depth,
                name=name,
                body=body,
                description=description,
                language=document_utils.detect_language(body), # 言語検出
                vectorstore_path=None, # 初期挿入時は None
                dbcon=None, # insert=False なので不要
                insert=False, # ここではDB挿入しない
                vectorize=False # ここではベクトル化しない
            )
            paragraph_objects.append(para_obj)
            order_counter += 1

            # 現在の深さの親IDを更新
            parent_id_map[depth] = para_obj.id # ここではまだ para_obj.id は None
            # 挿入後に parent_id_map を更新する必要がある -> ロジックの見直しが必要

        # 段落リストをDBに挿入し、IDを取得
        # insert_all は insert=True を前提としたループではないため、各オブジェクトに対して insert を呼ぶ必要がある
        # あるいは insert_all を変更して、オブジェクトリストを受け取り、内部で各obj.insert()を呼ぶようにする
        # ここでは Paragraph.insert_all を変更する
        Paragraph.insert_all(conn, paragraph_objects, vectorize=False) # まず挿入のみ実行

        # 挿入後、オブジェクトにIDが設定されているので、vectorstore_pathを設定し、ベクトル化を実行、DBを更新する
        print("\n--- ベクトルストアパス設定とベクトル化 ---")
        vectorize_count = 0
        failed_vectorize_count = 0
        for i, para_obj in enumerate(paragraph_objects):
             if para_obj.id is not None and document.vectorstore_dir:
                  # ID を使って一意なパスを生成
                  para_obj.vectorstore_path = str(Path(document.vectorstore_dir) / f"para_{para_obj.id}.vect")
                  
                  # 親子関係のIDを解決 (この段階でしかIDが確定しないため)
                  # parent_id_map を挿入前ではなく挿入後に構築し直すか、別の方法が必要
                  # 単純な連番順なら、前の段落IDを探すなどが考えられるが、構造が複雑だと難しい
                  # 一旦、parent_idの設定は init_paragraphs で行わず、別途階層構造を解析して設定する関数を用意するか、
                  # Paragraph.__init__ で parent_id を受け取るのではなく、親子関係を示すリストなどを受け取るようにする
                  # 現状のparent_id_mapロジックは挿入前IDなので正しくない。一旦 parent_id の設定はスキップするか、別途実装とする。
                  # Paragraph の parent_id は DB 上の Paragraph ID を参照する。
                  # extract_paragraphs の結果は (depth, name, body, description) のリストであり、ID情報は持たない。
                  # 抽出順＝順序 なので、親になる可能性があるのは前の段落。
                  # 階層構造を考慮した parent_id の設定は extract_paragraphs の実装と連携する必要がある。
                  # 例: depth 2 の親は直前の depth 1 の段落。depth 3 の親は直前の depth 2 の段落。
                  # extract_paragraphs が構造を保ったリストを返すなら、それを元に親IDを計算できる。
                  # parent_id_map を実際の Paragraph ID を保存するように修正する。

                  # 挿入後、parent_id を解決して設定
                  # 抽出された構造 (depth, ...) リストのインデックスと、paragraph_objects リストのインデックスは一致する
                  # 抽出元の structured_paragraphs リストを保持しておく
                  current_depth = structured_paragraphs[i][0]
                  current_para_id = para_obj.id # この段落のID
                  parent_id = None
                  
                  # 現在の段落より前の段落を逆順に探す
                  for j in range(i - 1, -1, -1):
                      prev_item = structured_paragraphs[j]
                      prev_para_obj = paragraph_objects[j]
                      # 親は現在の段落よりdepthが一つ浅い直近の段落
                      if prev_item[0] == current_depth - 1:
                          parent_id = prev_para_obj.id
                          break # 見つかったら終了

                  para_obj.parent_id = parent_id # 親IDを設定

                  # vectorstore_path と parent_id を更新するためDBを更新
                  try:
                       para_obj.update(conn)
                       # print(f"  {i+1}/{len(paragraph_objects)}: ✅ ID:{para_obj.id} - vectorstore_path / parent_id 更新")
                  except Exception as e:
                       print(f"  {i+1}/{len(paragraph_objects)}: ❌ ID:{para_obj.id} - vectorstore_path / parent_id 更新失敗: {e}")
                       # 失敗してもベクトル化に進むか、スキップするか？ スキップが安全。
                       failed_vectorize_count += 1 # 更新失敗もベクトル化失敗とみなす
                       continue

                  # ベクトル化を実行 (vectorize=Trueの場合)
                  if vectorize and embedding_name:
                      success = para_obj.vectorization(conn, embedding_name=embedding_name)
                      if success:
                           vectorize_count += 1
                           print(f"  {i+1}/{len(paragraph_objects)}: ✅ ID:{para_obj.id}, Name:'{para_obj.name}' - ベクトル化成功")
                      else:
                           failed_vectorize_count += 1
                           print(f"  {i+1}/{len(paragraph_objects)}: ❌ ID:{para_obj.id}, Name:'{para_obj.name}' - ベクトル化失敗")
                  else:
                       # ベクトル化しない場合でも、更新は行った
                       pass

        print(f"--- 段落初期化完了: 抽出 {len(paragraph_objects)} 件, ベクトル化成功 {vectorize_count} 件, ベクトル化失敗 {failed_vectorize_count} 件 ---")

    except Exception as e:
        print(f"❌ 段落初期化中にエラーが発生しました: {e}")
        # traceback.print_exc() # 詳細なトレースバックが必要なら有効化
        conn.rollback() # エラー時はロールバック
        raise e # エラーを再発生させる

def get_documents_by_project_id(conn: sqlite3.Connection, project_id: int) -> List["Document"]:
    """
    指定されたプロジェクトIDに紐づくDocumentインスタンスをDBから取得する。

    Parameters:
        conn: SQLiteの接続オブジェクト
        project_id: 取得したい文書が属するプロジェクトのID

    Returns:
        Documentインスタンスのリスト。エラー時や該当がない場合は空リスト。
    """
    cur = conn.cursor()
    documents = []
    try:
        # Document.from_row が期待するカラム順序でSELECTする
        cur.execute("""
            SELECT id, project_id, category_id, name, description, file_path, file_type, vectorstore_dir, embedding_model, status, created_at, updated_at
            FROM documents
            WHERE project_id = ?
            ORDER BY created_at DESC -- 作成日の新しい順にソート
        """, (project_id,))

        rows = cur.fetchall() # [(id, project_id, ...), ...] のタプルリスト
        documents = [Document.from_row(row) for row in rows]

        # Note: この関数でロードしたDocumentオブジェクトの paragraphs フィールドは空です。
        # 段落が必要な場合は、各Documentオブジェクトに対して別途 load_paragraphs(conn) を呼び出してください。

        return documents

    except sqlite3.Error as e:
        print(f"プロジェクトID {project_id} の文書リスト取得中にエラーが発生しました: {e}")
        return [] # エラー時は空リストを返す
    except ValueError as e:
         # from_row が失敗した場合など
         print(f"プロジェクトID {project_id} の文書データ形式エラー: {e}")
         return []
    except Exception as e:
        # その他の予期せぬエラー
        print(f"プロジェクトID {project_id} の文書リスト取得中に予期せぬエラーが発生しました: {e}")
        return []
