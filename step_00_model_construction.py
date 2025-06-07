import os
import csv
import sys
from pathlib import Path
import inspect
import json
import sqlite3

from core import document_utils as du # type: ignore
from core.objects import ( # type: ignore
    database,
    categories, 
    projects, 
    documents, 
    paragraphs
 )

# --- ディレクトリ設定 ---
# 現在のファイル（スクリプト）のパス
current_path = Path(__file__).resolve()

# 'core' ディレクトリを含む親ディレクトリを見つける
core_root = next(p for p in current_path.parents if p.name == "local-SLM-library")

# そこから目的のサブパスを定義
sample_dir = core_root / "sample"

# データベースのパス
db_dir = core_root / "database"
db_path = db_dir / "scenario.db"

markdown_dir = db_dir / "markdown"
vectorstore_dir = db_dir / "vectorstore"
scenario_dir = sample_dir / "scenario"

# Stage定義のCSVファイル名
stage_file_csv = scenario_dir / "stage.csv" # Adjust path if needed
stage_file_json = scenario_dir / "stage.json" # Adjust path if needed


def define_categories(conn, parent_ids=None, list_categories: list[str]=None) -> list[categories.Category]:
    """
    データベースにステージングに関連するカテゴリを定義します。

    Args:
        conn: データベース接続オブジェクト。

    Returns:
        list[category.Category]: 作成されたカテゴリのリスト。
    """
    new_categories = {}
    if parent_ids is not None:
        cat_paret_ids = parent_ids

    # 各ステージのカテゴリを作成
    for label in list_categories:
        cat = categories.Category(
            name=label,
            description=f"{label} に関するカテゴリー",
            parent_ids= cat_paret_ids,
            type_code="hier",
            sort_order=len(new_categories) + 1,
            dbcon=conn, 
            insert=True
        )
        new_categories[label]=cat
    return new_categories

def vectorstore_construction(
        dbconn:sqlite3.Connection, 
        project:projects.Project, 
        root_category:categories.Category,
        json_file_path:Path, 
        vectorstore_dir: Path, 
        markdown_dir: Path,
        entry_dict: list[str]
    ) -> list[dict]:
    """
    指定されたJSONファイルからデータを読み込み、Stageクラスのオブジェクトのリストとして返します。

    Args:
        json_file_path (str): 読み込むJSONファイルへのパス。

    Returns:
        list[Stage]: Stageクラスのオブジェクトのリスト。
                     ファイルの読み込みや処理に失敗した場合は空のリストを返します。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSONのデコード中にエラーが発生しました: {e}")
        return []
    except Exception as e:
        print(f"ファイルの読み込み中に予期しないエラーが発生しました: {e}")
        return []
    
    stage_category = define_categories(dbconn, [root_category.id], list(data.keys()))

    result = {}

    for stage in data:
        vectorstore_dir_stage = vectorstore_dir / stage
        markdown_path = markdown_dir / f"{stage}.md"

        # ベクトルストアとマークダウンのディレクトリが存在しない場合は作成
        if not vectorstore_dir_stage.exists():
            print(f"Vectorstore directory does not exist for stage {stage}. Creating it.")
            os.makedirs(vectorstore_dir_stage, exist_ok=True)
        if not markdown_dir.exists():
            print(f"Markdown directory does not exist for stage {stage}. Creating it.")
            os.makedirs(markdown_dir, exist_ok=True)

        explanation_category = define_categories(dbconn, [stage_category[stage].id], list(entry_dict.keys()))

        print(markdown_path)

        new_document = documents.Document(
            project_id = new_project.id,
            category_id = stage_category[stage].id, # 文書全体に紐づくカテゴリID
            name = stage,
            description=f"Expanation for {stage}", # 説明を追加
            file_path = markdown_path,
            vectorstore_dir = vectorstore_dir_stage, # 文書ごとのベクトルストア格納ディレクトリ
            embedding_model = new_project.default_embedding, # プロジェクトのデフォルトを使用
            dbcon=dbconn,
            insert=True, # Document を挿入し、init_paragraphs を呼び出す
        )

        content_lines = []
        content_lines.append(f"# {stage}")
        content_lines.append("")

        for key,label in entry_dict.items():
            explanation_text =data[stage][key]

            content_lines.append(f"## {key}")
            content_lines.append("")
            content_lines.append(f"### {label}")
            content_lines.append("")
            content_lines.append(explanation_text)
            content_lines.append("")

            new_paragraph = paragraphs.Paragraph(
                document_id = new_document.id,
                parent_id=None,
                category_id = explanation_category[key].id, # 文書全体に紐づくカテゴリID
                order=0,
                depth=0, 
                name = key,
                body= explanation_text, # 説明を追加
                description=f"Content from {markdown_path.name}", # 説明を追加
                language="ja", # 日本語を指定
                vectorstore_path = vectorstore_dir_stage / label, # これから作成するベクトルストアのパス
                dbcon=dbconn,
                insert=True, # Paragraph を挿入し、init_paragraphs を呼び出す
                vectorize= True, # ベクトル化を有効にする
                embedding_name= new_project.default_embedding, # プロジェクトのデフォルトを使用
            )
            new_document.paragraphs.append(new_paragraph)
        try:
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content_lines))
            print(f"'{markdown_path}' を正常に生成しました。")
        except IOError as e:
            print(f"ファイル '{markdown_path}' の書き込み中にエラーが発生しました: {e}")
        
        result[stage] = {
            "document": new_document,
            "paragraphs": new_document.paragraphs,
            "vectorstore_dir": vectorstore_dir_stage,
            "markdown_path": markdown_path
        }

    return result

if __name__ == "__main__":
    print("--- Running Stage Definition Script ---")

    # Step 1: データベースを構築する
    database.init_db(db_path, overwrite=True)
    conn = database.db_connect(db_path)
    if conn is None:
        print("Error: Could not connect to the database.")
        sys.exit(1)
    print("✅ Database initialized successfully.")


    # Step 2: プロジェクトとカテゴリを作成する
    # ステージングの親カテゴリを作成
    new_project = projects.Project(
        name="Decision Support for Pancreatic Cancer Staging",
        description="膵がんのステージングに関する情報を提供し、患者の診断と治療計画を支援するためのプロジェクト。",
        author="藤本悠",
        status="active",
        default_model_name="granite3.3:8b",
        default_prompt_name="japanese_concise",
        default_embedding_name="bge-m3",
        notes="実装のテスト",
        dbcon=conn, 
        insert=True
    )

    # ステージングの親カテゴリを作成
    cat_root = categories.Category(
        name="ステージング",
        description="膵がんのステージングに関するカテゴリー",
        type_code="hier",
        sort_order=0,
        dbcon=conn, 
        insert=True
    )

    entry_dict = {
        "検査結果":"test_results", 
        "自覚症状":"symptoms", 
        "生活習慣":"lifestyle", 
        "既往歴":"medical_history", 
        "家族歴":"family_history"
    }

    # ステップ 3: ステージングの定義ファイル（JSON）を読み込み、ステージングのためのベクトルハウスを構築する
    if not stage_file_json.exists():
        print(f"Error: Sample JSON file not found at {stage_file_json}")
        print("Please place a sample .json file in the ./sample/scenario directory or update the file_path.")
        sys.exit(1)
    # When the script is executed directly, call the main function

    # ステップ 4: ステージングのベクトルストアを構築する
    vectorstores = vectorstore_construction(
        conn, 
        new_project, 
        cat_root,
        stage_file_json, 
        vectorstore_dir, 
        markdown_dir,
        entry_dict
    )
    if not vectorstores:
        print("Error: No vectorstores were created.")
        sys.exit(1)
    
    print("✅ Vectorstore construction completed successfully.")
    print("Vectorstore directories and markdown files have been created successfully.")
    print("Vectorstore directories:")
    
    for stage, data in vectorstores.items():
        print(f"  - {stage}: {data['vectorstore_dir']}")
        print(f"  - Markdown file: {data['markdown_path']}")
        for paragraph in data['paragraphs']:
            print(f"    - Paragraph: {paragraph.name} (ID: {paragraph.id})")

    print("Stage definitions have been successfully created and stored in the database.")
    print("You can now use these vectorstores for further processing or querying.")
    print("Script completed successfully.")

