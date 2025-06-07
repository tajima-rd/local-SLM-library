import sqlite3
from typing import Optional, List, Any, Tuple, Dict
from dataclasses import dataclass, field

# === 抽象クラス定義 ===
from core import retriever_utils  # 相対インポートに変更
from .database import DBObject  # 相対インポートに変更

@dataclass(init=False)
class Category(DBObject):
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    parent_ids: List[int] = field(default_factory=list) 
    type_code: str = "hier"
    sort_order: int = 0
    level: int = field(default=0)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        parent_ids: Optional[List[int]] = None,
        type_code: str = "hier",
        sort_order: int = 0,
        dbcon: Optional[sqlite3.Connection] = None,
        insert: bool = True,
    ):
        self.id = None
        self.name = name
        self.description = description
        self.parent_ids = [pid for pid in (parent_ids if parent_ids is not None else []) if isinstance(pid, int) and pid is not None]
        self.type_code = type_code
        self.sort_order = sort_order
        self.level = 0 
        self.created_at = None 
        self.updated_at = None 

        if insert:
            if dbcon is None:
                raise ValueError("insert=True の場合、dbcon を指定する必要があります。")
            
            # レベルの計算
            if self.type_code == "hier" and self.parent_ids:
                 # 親カテゴリのレベルを取得
                 max_parent_level = -1
                 cur = dbcon.cursor()
                 parent_ids_tuple = tuple(self.parent_ids) # SELECT IN 用にタプル化
                 try:
                      cur.execute(f"""
                           SELECT MAX(level) FROM {self.table_name()}
                           WHERE id IN ({','.join('?' for _ in parent_ids_tuple)})
                      """, parent_ids_tuple)
                      result = cur.fetchone()
                      if result and result[0] is not None:
                           max_parent_level = result[0]
                 except sqlite3.Error as e:
                      print(f"親カテゴリのレベル取得中にエラーが発生しました: {e}")
                      # エラー時はレベル0として続行
                      max_parent_level = -1

                 self.level = max_parent_level + 1 if max_parent_level >= 0 else 0
            else:
                 # 非階層型または親がいない階層型はレベル0
                 self.level = 0
            # insert メソッドを呼び出す (計算したレベルが使われる)
            self.insert(dbcon)

    def insert(self, conn: sqlite3.Connection) -> int:
        """
        カテゴリ情報をcategoriesテーブルとcategory_parentsテーブルに挿入する。
        levelは__init__で計算された値、またはデフォルト値(0)が使用される。
        注意：挿入されたカテゴリの子孫カテゴリのlevelは自動更新されない。
        """
        cur = conn.cursor()
        # categories テーブルへの挿入 (level カラムを含む)
        # UNIQUE制約があるため、重複する名前の場合はエラーになる
        try:
            cur.execute('''
                INSERT INTO categories (name, description, type_code, sort_order, level, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''', (self.name, self.description, self.type_code, self.sort_order, self.level)) # self.level を追加

            self.id = cur.lastrowid # 挿入されたカテゴリのIDを取得

            # 中間テーブル category_parents への挿入
            # parent_ids は __init__ で int にフィルタリング済み
            if self.id is not None and self.parent_ids:
                # 重複を避けるために set を使う
                links_to_insert = [(self.id, parent_id) for parent_id in set(self.parent_ids) if isinstance(parent_id, int) and parent_id is not None]
                if links_to_insert:
                     # INSERT OR IGNORE を使うと、既に存在する親子関係は無視される
                     cur.executemany('''
                         INSERT OR IGNORE INTO category_parents (child_category_id, parent_category_id)
                         VALUES (?, ?)
                     ''', links_to_insert)

            conn.commit()
            # オブジェクトのlevelは__init__で計算済みだが、DB上の子孫levelは古いまま
            return self.id if self.id is not None else -1
        except sqlite3.IntegrityError as e:
             conn.rollback() # エラー時はロールバック
             # 重複エラーの場合は、既存のカテゴリIDを返すなどの処理も考えられるが、
             # ここではシンプルに例外を再発生させる
             raise e
        except sqlite3.Error as e:
             conn.rollback()
             print(f"データベース挿入中にエラーが発生しました: {e}")
             raise e

    def update(self, conn: sqlite3.Connection):
        """
        カテゴリ情報をcategoriesテーブルとcategory_parentsテーブルで更新する。
        親子の変更があった場合、自身のlevelを再計算してDBを更新する。
        注意：子孫カテゴリのlevelは自動更新されない。
        """
        if self.id is None:
            raise ValueError("IDがないCategoryオブジェクトは更新できません。")

        cur = conn.cursor()

        # 更新前の parent_ids を取得（レベル再計算が必要か判断するため）
        old_parent_ids = []
        try:
            cur.execute("""
                SELECT parent_category_id
                FROM category_parents
                WHERE child_category_id = ?
            """, (self.id,))
            old_parent_rows = cur.fetchall()
            old_parent_ids = [p[0] for p in old_parent_rows if p and isinstance(p[0], int) and p[0] is not None]
        except sqlite3.Error as e:
             print(f"更新前の親カテゴリ情報取得中にエラーが発生しました: {e}")
             # エラー時は親が変わったとみなしてレベル再計算を試みる

        # 新しい parent_ids に基づいて自身のレベルを再計算
        new_level = 0
        if self.type_code == "hier" and self.parent_ids:
             max_parent_level = -1
             parent_ids_tuple = tuple(self.parent_ids)
             try:
                  cur.execute(f"""
                       SELECT MAX(level) FROM {self.table_name()}
                       WHERE id IN ({','.join('?' for _ in parent_ids_tuple)})
                  """, parent_ids_tuple)
                  result = cur.fetchone()
                  if result and result[0] is not None:
                       max_parent_level = result[0]
             except sqlite3.Error as e:
                  print(f"更新時の親カテゴリのレベル取得中にエラーが発生しました: {e}")
                  # エラー時はレベル0として続行
                  max_parent_level = -1
             new_level = max_parent_level + 1 if max_parent_level >= 0 else 0
        else:
             # 非階層型または親がいない階層型はレベル0
             new_level = 0

        # カテゴリオブジェクトの level 属性を更新後の値にする
        self.level = new_level

        # categories テーブルの更新 (level カラムを含む)
        # UNIQUE制約があるため、重複する名前の場合はエラーになる可能性がある
        try:
            cur.execute('''
                UPDATE categories SET name=?, description=?, type_code=?, sort_order=?, level=?, updated_at=datetime('now')
                WHERE id=?
            ''', (self.name, self.description, self.type_code, self.sort_order, self.level, self.id)) # self.level を追加

            # 中間テーブル category_parents の更新
            # 既存の親リンクを全て削除
            cur.execute('DELETE FROM category_parents WHERE child_category_id=?', (self.id,))

            # 新しい親リンクを挿入
            # parent_ids は __init__ またはロード時に設定されたリスト
            if self.parent_ids:
                # 重複を避けるために set を使う
                 # parent_id は int または None が parent_ids に含まれる可能性を考慮
                links_to_insert = [(self.id, parent_id) for parent_id in set(self.parent_ids) if isinstance(parent_id, int) and parent_id is not None]
                if links_to_insert:
                    # INSERT OR IGNORE を使うと、既に存在する親子関係は無視される
                    cur.executemany('''
                        INSERT OR IGNORE INTO category_parents (child_category_id, parent_category_id)
                        VALUES (?, ?)
                    ''', links_to_insert)

            conn.commit()

            # 注意: ここで self.id の level は更新されたが、このカテゴリの子孫の level は古いままです。
            # 子孫の level を連鎖的に更新するロジックは別途必要です。

        except sqlite3.IntegrityError as e:
             conn.rollback() # エラー時はロールバック
             raise e # エラーを再発生させる
        except sqlite3.Error as e:
             conn.rollback()
             print(f"データベース更新中にエラーが発生しました: {e}")
             raise e

    def delete(self, conn: sqlite3.Connection):
        """
        指定されたIDのカテゴリをDBから削除する。
        中間テーブルの関連エントリはCASCADEにより自動削除される。
        """
        if self.id is None:
            raise ValueError("IDがないCategoryオブジェクトは削除できません。")
        cur = conn.cursor()
        try:
            cur.execute('DELETE FROM categories WHERE id=?', (self.id,))
            # Note: related entries in category_parents are deleted by CASCADE
            # 削除されたカテゴリの子孫カテゴリも CASCADE で削除される場合、レベルの連鎖更新は不要
            # documents, paragraphs との FOREIGN KEY 設定 (ON DELETE SET NULL) もレベルには影響しない
            conn.commit()
        except sqlite3.Error as e:
             conn.rollback() # エラー時はロールバック
             print(f"データベース削除中にエラーが発生しました: {e}")
             raise e

    def to_retriever(self, conn: sqlite3.Connection) -> retriever_utils.RetrieverCategory:
        try:
             # 型ヒントのために RetrieverCategory クラスをインポートする必要がある
             # ここで import すると循環参照になる可能性があるため、型ヒントは文字列で行い、
             # 実際のクラス利用時はグローバルスコープや別の方法で参照する設計が必要
             # 例: `from core import retriever_utils` をファイルの冒頭に置き、
             #     そのモジュールが循環参照にならないように core の設計を確認する
             # ここでは import が成功するものとして、retriever_utils を直接使用
             pass
        except ImportError:
             raise ImportError("core.retriever_utils がインポートできません。RetrieverCategory への変換に失敗しました。")

        if self.type_code == "hier":
            # RetrieverUtils側の HierarchicalRetrieverCategory は parent_ids (List[str]) を持つ設計
            # DB側が複数親を許容しているため、Retriever側へのマッピング方法を決定する必要がある
            # ここでは、複数親があっても parent_ids リストに含めるが、警告を出す。
            # Retriever側がList[str]を受け付ける前提。

            parent_tagnames: List[str] = [] # 親カテゴリ名 list(str)
            for parent_id in self.parent_ids:
                 # Category.get_by_id に conn を渡す
                 # get_by_id はDBから level を読み込むが、to_retriever_category では level は使用しないため問題なし
                 parent_cat = Category.get_by_id(conn=conn, category_id=parent_id)
                 if parent_cat:
                      parent_tagnames.append(parent_cat.name)
                 else:
                      # 親カテゴリが見つからない場合は警告を出すが、リストには追加しない
                      print(f"⚠️ 親カテゴリID {parent_id} が見つかりません (子カテゴリ: '{self.name}', ID: {self.id})。この親は RetrieverCategory の parent_ids に含められません。")

            # RetrieverUtils側のHierarchicalRetrieverCategoryのparent_idsはList[str]である想定
            return retriever_utils.HierarchicalRetrieverCategory( # retriever_utils を直接使用
                tagname=self.name,
                parent_ids=parent_tagnames, # 親カテゴリ名 list(str) を渡す
                level=self.level # levelはDBから読み込まれた値を使用
            )

        elif self.type_code == "flat":
            # RetrieverUtils側に FlatRetrieverCategory が定義されていることを前提とする
            return retriever_utils.FlatRetrieverCategory( # retriever_utils を直接使用
                tagname=self.name
            )

        elif self.type_code == "array":
             # TODO: RetrieverUtils側に ArrayRetrieverCategory の実装
             raise NotImplementedError("配列型カテゴリのRetrieverCategory変換は未実装です。")
        else:
            # 未知の type_code の場合はエラーとする
            raise ValueError(f"不明な type_code '{self.type_code}' のカテゴリ '{self.name}' (ID: {self.id})。RetrieverCategoryへの変換に失敗しました。")

    @classmethod
    def table_name(cls) -> str:
        return "categories"

    @classmethod
    def from_row(cls, row: Tuple[Any]) -> 'Category':
        # DBから読み込むカラムと Category オブジェクトの属性をマッピング
        # row は id, name, description, type_code, sort_order, level, created_at, updated_at の順を想定
        if len(row) != 8: # 8カラムを想定
             raise ValueError(f"Expected 8 columns for Category.from_row based on DB schema, got {len(row)}. Row: {row}")

        cat = cls.__new__(cls) # __init__ をスキップしてインスタンスを生成
        cat.id = row[0]
        cat.name = row[1]
        cat.description = row[2]
        # from_row 時点では親情報は中間テーブルにあるためロードしない
        cat.parent_ids = []
        cat.type_code = row[3]
        cat.sort_order = row[4]
        cat.level = row[5] # DBから level を読み込む
        cat.created_at = row[6]
        cat.updated_at = row[7]

        # 注意: この時点では parent_ids は空リストです。
        # 親情報のロードは get_by_id や get_all で行われます。
        # level はDBから読み込まれていますが、親子の変更によって古い値の可能性があります。

        return cat

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list["Category"]:
        """
        全てのカテゴリをDBから取得し、親子関係を設定して返す。levelはDBから読み込まれる。
        注意：DB上のlevel値は、親子の変更によって古い可能性がある。
        """
        cur = conn.cursor()
        # level カラムを含む全てのカラムを取得
        cur.execute("""
            SELECT id, name, description, type_code, sort_order, level, created_at, updated_at
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
        parent_map: Dict[int, List[int]] = {}
        for child_id, parent_id in parent_links:
            if child_id not in parent_map:
                parent_map[child_id] = []
            if isinstance(parent_id, int) and parent_id is not None:
                 parent_map[child_id].append(parent_id)

        # Category オブジェクトを生成し、親リンク情報を設定
        categories: List[Category] = []
        for row in category_rows:
            # from_row が level を含む8カラムを想定するように修正済み
            cat = cls.from_row(row)
            if cat.id is not None:
                # 中間テーブルから取得した親IDリストを設定
                cat.parent_ids = parent_map.get(cat.id, [])
                # level は from_row でDBから読み込まれている
                categories.append(cat)

        return categories

    @classmethod
    def get_by_id(cls, conn: sqlite3.Connection, category_id: int) -> Optional["Category"]:
        """
        指定されたIDのCategoryインスタンスをDBから取得し、親カテゴリ情報をロードする。levelはDBから読み込まれる。
        注意：DB上のlevel値は、親子の変更によって古い可能性がある。

        Parameters:
            conn: SQLiteの接続オブジェクト
            category_id: 取得したいカテゴリID

        Returns:
            Categoryインスタンス または None（該当がない場合）
        """
        cur = conn.cursor()
        # level カラムを含む全てのカラムを取得
        cur.execute(f'''
            SELECT id, name, description, type_code, sort_order, level, created_at, updated_at
            FROM {cls.table_name()} WHERE id=?
        ''', (category_id,))
        row = cur.fetchone()

        if row:
            # from_row が level を含む8カラムを想定するように修正済み
            cat = cls.from_row(row) # 基本情報+levelでオブジェクトを作成

            # 中間テーブルから親IDリストを取得
            cur.execute("""
                SELECT parent_category_id
                FROM category_parents
                WHERE child_category_id = ?
            """, (category_id,))
            parent_rows = cur.fetchall()
            # 親IDリストを設定 (Noneを除外し、int型であることを確認)
            cat.parent_ids = [p[0] for p in parent_rows if p and isinstance(p[0], int) and p[0] is not None]

            # level は from_row でDBから読み込まれているため、ここでは設定不要。
            # ただし、DBの値が古い可能性がある点に注意。

            return cat
        else:
            return None

    @classmethod
    def get_by_level(cls, conn: sqlite3.Connection, level: int) -> List["Category"]:
        """
        指定されたレベルのCategoryインスタンスリストをDBから取得し、親カテゴリ情報もロードする。
        levelはDBから読み込まれる。
        注意：DB上のlevel値は、親子の変更によって古い可能性がある。

        Parameters:
            conn: SQLiteの接続オブジェクト
            level: 取得したいカテゴリの階層レベル

        Returns:
            一致するCategoryインスタンスのリスト。該当がない場合は空リスト。
        """
        cur = conn.cursor()
        # level カラムを含む全てのカラムを取得
        cur.execute(f'''
            SELECT id, name, description, type_code, sort_order, level, created_at, updated_at
            FROM {cls.table_name()} WHERE level=?
        ''', (level,))
        category_rows = cur.fetchall() # fetchone() を fetchall() に変更

        if not category_rows:
            return [] # 該当するカテゴリがなければ空リストを返す

        # 取得したカテゴリIDのリストを作成
        category_ids = [row[0] for row in category_rows if row[0] is not None]
        if not category_ids:
             return [] # IDがNoneの行は無視

        # 中間テーブルから、取得したカテゴリに関連する全ての親子関係を取得
        # SELECT IN 句を使用して効率的に取得
        category_ids_tuple = tuple(category_ids)
        cur.execute(f"""
            SELECT child_category_id, parent_category_id
            FROM category_parents
            WHERE child_category_id IN ({','.join('?' for _ in category_ids_tuple)})
        """, category_ids_tuple)
        parent_links = cur.fetchall()

        # 親リンク情報を、子カテゴリIDをキーとする辞書に整理
        # Dict を使うために typing から Dict をインポートする必要があるかもしれません (既存コードでListなどと共にインポートされているか確認ください)
        parent_map: Dict[int, List[int]] = {} # Dict を使用
        for child_id, parent_id in parent_links:
            if child_id not in parent_map:
                parent_map[child_id] = []
            # Noneを除外し、int型であることを確認してリストに追加
            if isinstance(parent_id, int) and parent_id is not None:
                 parent_map[child_id].append(parent_id)

        # Category オブジェクトを生成し、親リンク情報を設定
        categories: List["Category"] = []
        for row in category_rows:
            # from_row が level を含む8カラムを想定するように修正済み
            cat = cls.from_row(row)
            if cat.id is not None:
                # 中間テーブルから取得した親IDリストを設定
                cat.parent_ids = parent_map.get(cat.id, [])
                # level は from_row でDBから読み込まれている
                categories.append(cat)

        return categories # Categoryオブジェクトのリストを返す

    @classmethod
    def get_by_name(cls, conn: sqlite3.Connection, name: str) -> List["Category"]:
        """
        指定された名前のCategoryインスタンスリストをDBから取得し、親カテゴリ情報をロードする。
        同じ名前のカテゴリが複数存在する可能性があるため、リストで返す。
        levelはDBから読み込まれる。

        Parameters:
            conn: SQLiteの接続オブジェクト
            name: 取得したいカテゴリ名

        Returns:
            一致するCategoryインスタンスのリスト。該当がない場合は空リスト。
        """
        cur = conn.cursor()
        # level カラムを含む全てのカラムを取得
        cur.execute(f'''
            SELECT id, name, description, type_code, sort_order, level, created_at, updated_at
            FROM {cls.table_name()} WHERE name=?
        ''', (name,))
        category_rows = cur.fetchall()

        if not category_rows:
            return [] # 該当するカテゴリがなければ空リストを返す

        # 取得したカテゴリIDのリストを作成
        category_ids = [row[0] for row in category_rows if row[0] is not None]
        if not category_ids:
             return [] # IDがNoneの行は無視

        # 中間テーブルから、取得したカテゴリに関連する全ての親子関係を取得
        # SELECT IN 句を使用して効率的に取得
        category_ids_tuple = tuple(category_ids)
        cur.execute(f"""
            SELECT child_category_id, parent_category_id
            FROM category_parents
            WHERE child_category_id IN ({','.join('?' for _ in category_ids_tuple)})
        """, category_ids_tuple)
        parent_links = cur.fetchall()

        # 親リンク情報を、子カテゴリIDをキーとする辞書に整理
        parent_map: Dict[int, List[int]] = {}
        for child_id, parent_id in parent_links:
            if child_id not in parent_map:
                parent_map[child_id] = []
            if isinstance(parent_id, int) and parent_id is not None:
                 parent_map[child_id].append(parent_id)

        # Category オブジェクトを生成し、親リンク情報を設定
        categories: List[Category] = []
        for row in category_rows:
            # from_row が level を含む8カラムを想定するように修正済み
            cat = cls.from_row(row)
            if cat.id is not None:
                # 中間テーブルから取得した親IDリストを設定
                cat.parent_ids = parent_map.get(cat.id, [])
                # level は from_row でDBから読み込まれている
                categories.append(cat)

        return categories
    
    @staticmethod
    def get_children(conn: sqlite3.Connection, category_id: int) -> List["Category"]:
        """
        指定されたカテゴリの直接の子カテゴリのリストをDBから取得し、親カテゴリ情報もロードする。
        levelはDBから読み込まれる。
        注意：DB上のlevel値は、親子の変更によって古い可能性がある。

        Parameters:
            conn: SQLiteの接続オブジェクト
            category_id: 子カテゴリを取得したい親カテゴリのID

        Returns:
            直接の子カテゴリのCategoryインスタンスのリスト。該当がない場合は空リスト。
        """
        cur = conn.cursor()
        child_ids: List[int] = []

        try:
            # 1. 指定された category_id を親として持つ子カテゴリの ID を category_parents から取得
            cur.execute("""
                SELECT child_category_id
                FROM category_parents
                WHERE parent_category_id = ?
            """, (category_id,))
            child_rows = cur.fetchall()
            # Noneを除外し、int型であることを確認してリストに追加
            child_ids = [row[0] for row in child_rows if row and isinstance(row[0], int) and row[0] is not None]

            if not child_ids:
                return [] # 子カテゴリがなければ空リストを返す

            # 2. 取得した子カテゴリ ID のリストを使って、categories テーブルからまとめて子カテゴリ情報を取得
            # level カラムを含む全てのカラムを取得
            # SELECT IN 句用にタプルに変換
            child_ids_tuple = tuple(child_ids)
            cur.execute(f"""
                SELECT id, name, description, type_code, sort_order, level, created_at, updated_at
                FROM categories
                WHERE id IN ({','.join('?' for _ in child_ids_tuple)})
            """, child_ids_tuple)
            category_rows = cur.fetchall()

            # 3. 子カテゴリに関連する全ての親子関係を category_parents から取得
            # SELECT IN 句を使用して効率的に取得
            cur.execute(f"""
                SELECT child_category_id, parent_category_id
                FROM category_parents
                WHERE child_category_id IN ({','.join('?' for _ in child_ids_tuple)})
            """, child_ids_tuple)
            parent_links = cur.fetchall()

            # 4. 親リンク情報を、子カテゴリIDをキーとする辞書に整理
            # Dict を使うために typing から Dict をインポートする必要があるかもしれません (既存コードでListなどと共にインポートされているか確認ください)
            parent_map: Dict[int, List[int]] = {}
            for child_id, parent_id in parent_links:
                if child_id not in parent_map:
                    parent_map[child_id] = []
                # Noneを除外し、int型であることを確認してリストに追加
                if isinstance(parent_id, int) and parent_id is not None:
                     parent_map[child_id].append(parent_id)

            # 5. Category オブジェクトを生成し、親リンク情報を設定
            children: List["Category"] = []
            for row in category_rows:
                 # from_row が level を含む8カラムを想定するように修正済み
                 # DBObjectから継承したfrom_rowクラスメソッドを使用
                 cat = Category.from_row(row)
                 if cat.id is not None:
                     # 中間テーブルから取得した親IDリストを設定
                     cat.parent_ids = parent_map.get(cat.id, [])
                     # level は from_row でDBから読み込まれている
                     children.append(cat)

            return children

        except sqlite3.Error as e:
            print(f"Error getting children for category ID {category_id}: {e}")
            # エラー時は空リストを返すか、例外を再発生させるか。ここでは空リストを返す。
            return []
    
    @staticmethod
    def get_category_id_name_pairs(
        conn: sqlite3.Connection,
        type_code: Optional[str] = None
    ) -> List[Tuple[int, str]]:
        """
        指定された type_code のカテゴリの ID と名前のペアリストを取得する。
        UIのセレクタなどで利用することを想定。

        Parameters:
            conn: SQLiteの接続オブジェクト
            type_code: フィルタリングしたいカテゴリのタイプコード (例: 'hier', 'flat')。Noneの場合は全タイプ。

        Returns:
            [(category_id, category_name), ...] の形式のリスト。
        """
        cur = conn.cursor()

        sql = "SELECT id, name FROM categories"
        params = []

        if type_code is not None:
            sql += " WHERE type_code = ?"
            params.append(type_code)

        sql += " ORDER BY sort_order, name" # 順序と名前でソート

        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
            # fetchall() はデフォルトでタプルリストを返すので、そのまま返す
            return rows
        except sqlite3.Error as e:
            print(f"Error getting category id-name pairs: {e}")
            return []