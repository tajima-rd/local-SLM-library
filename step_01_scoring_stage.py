import os
import csv
import sys
from pathlib import Path
import inspect
import json
import sqlite3

from typing import Optional, List, Any, Tuple, Dict

from loader import load_project # type: ignore
from core import document_utils as du # type: ignore
from core import retriever_utils
from core.objects import ( 
    database,
    categories, 
    projects, 
    documents, 
    paragraphs
 )

# --- ディレクトリ設定 ---
current_path = Path(__file__).resolve()

# Let's stick to the logic from sample_03_staging.py to find the 'modules' dir itself
modules_root = next(p for p in current_path.parents if p.name == "local-SLM-library")

db_dir = modules_root / "database"
db_path = db_dir / "scenario.db"

vectorstore_dir = db_dir / "vectorstore"
scenario_dir = modules_root / "sample" / "scenario" # Or adjust path if sample is elsewhere

patiant_dir = modules_root / "sample" / "patient" # Directory for patient data

dialog_file_md = patiant_dir / "dialog.md" # Input dialog file
result_file_csv = patiant_dir / "result.csv" # Output results file

# How many top matches to find for each dialog chunk
TOP_K_MATCHES = 5
# Configuration for splitting the dialog text
DIALOG_CHUNK_SIZE = 4000
DIALOG_CHUNK_OVERLAP = 800

def get_category_name_from_id(conn: sqlite3.Connection, category_id: int) -> str:
    """Looks up a category name by its ID in the database."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM categories WHERE id = ?", (category_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        return f"Unknown Category ID: {category_id}"
    except Exception as e:
        print(f"Error querying database for category ID {category_id}: {e}")
        return f"DB Error for ID: {category_id}"


if __name__ == "__main__":
    # Step 1: Load the project and documents
    proj, docs = load_project(db_path, "Decision Support for Pancreatic Cancer Staging")

    # Step 2: Load and Split Dialog Document
    conn = database.db_connect(db_path)
    if conn is None:
        print(f"❌ Error: Could not connect to the database at {db_path}.")
        print("Please ensure sample_03_staging.py has been run successfully to create the database.")
        sys.exit(1)
    print(f"✅ Connected to database: {db_path}")

    if not dialog_file_md.exists():
        print(f"❌ Error: Dialog file not found at {dialog_file_md}")
        conn.close()
        sys.exit(1)

    dialog_chunk = None
    try:
        # Load the markdown file
        dialog_documents = du.load_documents(str(dialog_file_md), loader_type="markdown")
        if not dialog_documents:
             print(f"❌ Error: Could not load any content from {dialog_file_md}")
             conn.close()
             sys.exit(1)
        
        dialog_chunk = dialog_documents[0]  # Assuming single document in the file

    except Exception as e:
        print(f"❌ Error loading or splitting dialog file {dialog_file_md}: {e}")
        conn.close()
        sys.exit(1)
    
    # step 3:ステージングのためのカテゴリーを選ぶ
    # 階層がトップレベルのカテゴリーを取得
    print(f"--- Loading Categories ---")
    top_categories = categories.Category.get_by_level(conn, 0)

    if not top_categories:
        print(f"❌ Error: No top-level categories found in the database.")
        conn.close()
        sys.exit(1)
    else:
        for top_category in top_categories:
            print(f"ID {top_category.id} : {top_category.name}")

    # 今は暫定的に最初のトップレベルカテゴリーを選択
    category_children = categories.Category.get_children(conn, top_categories[0].id)
    first_descendants = categories.Category.get_children(conn, category_children[0].id)
    
    # step 4: 行と列のヘッダーの準備
    column_header = []
    row_header = []

    if not first_descendants:
        print(f"❌ Error: No first descendants found for top-level category {top_categories[0].name}.")
        conn.close()
        sys.exit(1)
    else:
        for entry in first_descendants:
            row_header.append(entry.name)

    if not category_children:
        print(f"❌ Error: No child categories found for top-level category {top_categories[0].name}.")
        conn.close()
        sys.exit(1)
    else:
        for child in category_children:
            column_header.append(child.name)

    # Step 5: Perform Similarity Search
    print(f"\n--- Perform Similarity Search ---")

    # 行列データを格納するための辞書を初期化
    # matrix_data[row_category_name][column_category_name] = score (float) or None
    # Step 4 で row_header, column_header が定義済みであることを前提とします。
    # column_header の最初の "." はデータとしては使用しないため、初期化ではスキップします。
    matrix_data: Dict[str, Dict[str, Optional[float]]] = {}
    for row_name in row_header: # テンプレートの行ヘッダー
        matrix_data[row_name] = {}
        for col_name in column_header: # テンプレートの列ヘッダー
            matrix_data[row_name][col_name] = float('inf') # 各セルを無限大で初期化

    scored_cell_count = 0 # スコアが格納されたセルのカウント (最終的な matrix_data のセル数)

    for doc in docs:
        doc_name = doc.name # ドキュメント名 (列ヘッダー名として使用)

        # ドキュメント名がテンプレートの有効な列ヘッダーに含まれているか確認
        if doc_name not in column_header:
             print(f"ℹ️ Skipping document '{doc_name}' as its name is not in the template column headers.")
             continue # スキップ


        print("====================") # 元コードに合わせる
        print(doc_name) # 元コードに合わせる

        # このドキュメント (列 doc_name) について、各行 (row_header のカテゴリー) の
        # スコア候補を保持する必要はない。matrix_data に直接書き込む。


        # 元コードの内部ループ: パラグラフごと
        for i, para in enumerate(doc.paragraphs):
            # para が紐づくカテゴリーを取得 (これがテンプレートの行ヘッダーに対応すると仮定)
            para_cat_id = para.category_id # 元コードにあった取得処理
            if para_cat_id is None:
                # print(f"  ℹ️ Paragraph {i+1} in '{doc.name}' has no category ID. Skipping.") # 詳細表示はコメントアウト
                continue

            # Category.get_by_id が使える前提
            para_cat = categories.Category.get_by_id(conn, para_cat_id)
            if para_cat is None:
                 print(f"  ℹ️ Paragraph {i+1} in '{doc.name}' has unknown category ID: {para_cat_id}. Skipping.")
                 continue

            row_name_for_para = para_cat.name # パラグラフのカテゴリー名 (行ヘッダー名として使用)

            # パラグラフのカテゴリー名がテンプレートの有効な行ヘッダーに含まれているか確認
            if row_name_for_para not in row_header:
                 print(f"  ℹ️ Paragraph {i+1} in '{doc.name}' (category '{row_name_for_para}') is not in the template row headers. Skipping search for this paragraph.")
                 continue # スキップ


            print(para.vectorstore_path) # 元コードに合わせる

            # ベクトルストアパスがあり、かつファイルが存在するか確認
            if not para.vectorstore_path or not Path(para.vectorstore_path).exists():
                print(f"  ℹ️ Paragraph has no vector store path or path does not exist. Skipping search.")
                continue

            # found_searchable_paragraph_in_doc = True # このフラグはドキュメント単位で持つ必要はない


            try:
                # Load the vector store from the directory
                vectorstore = retriever_utils.load_vectorstore(para.vectorstore_path)

                # Use similarity_search_with_score でスコア（距離）付きの検索結果を取得
                # k=TOP_K_MATCHES は元の値を使用
                results_with_scores = vectorstore.similarity_search_with_score(
                    dialog_chunk.page_content, # dialog_chunk.page_content は dialog_chunk_content に変更
                    k=TOP_K_MATCHES # 元の K 値を使用
                )

                # 検索結果を表示 (元のコードにあった print(results_with_scores) と同じ)
                print(results_with_scores)

                # 得られた結果の中から、最も距離が小さいスコア（類似度が高い）を見つける
                if results_with_scores:
                    best_score_this_paragraph_search = min(score for doc, score in results_with_scores)
                    # print(f"    Best score from this paragraph search: {best_score_this_paragraph_search:.4f}") # 詳細表示はコメントアウト

                    # このスコアを使って、matrix_data の該当セル (row_name_for_para, doc_name) の
                    # 現在の値と比較し、より小さいスコア（より類似度が高い）で更新
                    # row_name_for_para はパラグラフのカテゴリー名 (行ヘッダー)
                    # doc_name はドキュメント名 (列ヘッダー)
                    current_matrix_score_in_cell = matrix_data[row_name_for_para][doc_name] # 初期値または更新済みの値を取得

                    if best_score_this_paragraph_search < current_matrix_score_in_cell:
                         matrix_data[row_name_for_para][doc_name] = best_score_this_paragraph_search
                         # print(f"    Updated best score for cell [{row_name_for_para}][{doc_name}]: {matrix_data[row_name_for_para][doc_name]:.4f}") # 詳細表示


            except Exception as e:
                print(f"❌ Error searching vector store {para.vectorstore_path}: {e}")
                # エラーが発生しても処理を続行

    # Step 5 の最後で、matrix_data に有効なスコアが格納されたセルの数をカウント
    scored_cell_count = 0 # セルのカウントはループの外で合計
    for row_name in row_header:
        for col_name in column_header:
            score = matrix_data[row_name][col_name]
            if score is not None and score != float('inf'):
                 scored_cell_count += 1


    print(f"\n--- Scoring Complete. {scored_cell_count} cells scored in matrix. ---")

    # --- Step 6: 行列結果をCSVに出力 (テンプレートに合わせる) ---
    # matrix_data 変数に、行列に必要な結果が格納されていると仮定して処理します。
    print(f"\n--- Writing Matrix Results to CSV ---")

    # CSV書き込み用のデータリストを作成
    csv_output_rows: List[List[Any]] = []

    # ヘッダー行を追加: 最初のセルは "特徴の軸" + テンプレートの列ヘッダーリスト
    csv_output_rows.append(["."] + column_header)

    # データ行を追加: 行ヘッダー + 各列に対応するスコア
    # row_header はテンプレートの行ヘッダー
    for row_name in row_header:
        row_data: List[Any] = [row_name] # 行の最初は行ヘッダー名
        # column_header の順序でイテレートし、CSVの列順を決定
        for col_name in column_header: # col_name はテンプレートの列ヘッダー名 (doc.name)
            # matrix_data から対応するスコアを取得。
            score = matrix_data.get(row_name, {}).get(col_name)
            # スコアをリストに追加。None または float('inf') の場合は空文字列などを格納。
            if score is not None and score != float('inf'):
                row_data.append(f"{score:.4f}") # スコアを小数点以下4桁でフォーマット
            else:
                row_data.append("0.0") # テンプレートに合わせて 0.0 を格納

        csv_output_rows.append(row_data)


    # 出力ディレクトリが存在することを確認
    result_file_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        # データがなくてもヘッダー行はあるはずだが念のため
        if csv_output_rows:
            with open(result_file_csv, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(csv_output_rows) # 作成した行リスト全体を書き出し

            print(f"✅ Matrix results saved to {result_file_csv}")
        else:
            print(f"ℹ️ No matrix data generated. {result_file_csv} was not written.")


    except IOError as e:
        print(f"❌ Error writing matrix results to {result_file_csv}: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while writing matrix results: {e}")


    # Step 7: Close Database Connection
    conn.close()
    print("✅ Database connection closed.")
    print("--- Script Finished ---")