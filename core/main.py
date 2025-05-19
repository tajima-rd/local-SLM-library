# coding: utf-8

# 外部ライブラリから読み込んだモジュール類
import os
import ollama # type: ignore
from pathlib import Path
import database # type: ignore
from typing import Optional, List, Any, Tuple
import sqlite3


# 今回のプロジェクトのために開発した自作コード
from rag_session import RAGSession  # type: ignore
from retriever_utils import HierarchicalRetrieverCategory
import document_utils # type: ignore
import switch_rag_objects as sro

# --- ディレクトリ設定 ---
# 現在のファイル（スクリプト）のパス
current_path = Path(__file__).resolve()
 
# 'core' ディレクトリを含む親ディレクトリを見つける
core_root = next(p for p in current_path.parents if p.name == "core")

# そこから目的のサブパスを定義
base_dir = core_root / "sample"
markdown_dir = base_dir / "markdown"
vectorstore_dir = base_dir / "vectorstore"
pdf_dir = base_dir / "pdf"
db_path = base_dir / "database.db"


import re
from pathlib import Path

# 正規表現：第○章（空白対応・全角数字も対応）
CHAPTER_PATTERN = re.compile(r"^##\s*第\s*[一二三四五六七八九十0-9０-９]+\s*章")

# プレフィックスで階層をさらに下げるべきかを判定
PREFIX_DOWN_PATTERN = re.compile(r"^(###)\s*([ア-ン一二三四五六七八九十0-9０-９]+)")

def to_half_width_numbers(text: str) -> str:
    return text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

def is_chapter_heading(line: str) -> bool:
    normalized = to_half_width_numbers(line)
    return CHAPTER_PATTERN.match(normalized) is not None

def should_indent_further(line: str) -> bool:
    """カタカナ or 数字で始まるタイトルか？"""
    return bool(PREFIX_DOWN_PATTERN.match(line))

def collect_markdown_headings(md_path: Path) -> list[str]:
    """
    Markdownの'##'見出し行をすべて収集し、階層を調整：
    - 最初の見出しはそのまま
    - 「第○章」が含まれる場合は階層を戻して'##'
    - 通常は'###'
    - ただし、'###' で始まり、かつプレフィックスがカタカナ/数字なら '####'
    """
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    headings = [line.strip() for line in lines if line.strip().startswith("##")]

    adjusted = []
    for i, line in enumerate(headings):
        if i == 0:
            adjusted.append(line)  # 最初の見出しはそのまま
        elif is_chapter_heading(line):
            adjusted.append(re.sub(r"^##", "##", line, count=1))  # 章タイトルはそのまま
        else:
            if should_indent_further(line):  # カタカナ・数字なら1段下げ
                new_line = re.sub(r"^##", "###", line, count=1)
            else:  # 通常は2段下げ
                new_line = re.sub(r"^##", "####", line, count=1)
            adjusted.append(new_line)

    return adjusted




def get_category_path_by_question(question: str, conn: sqlite3.Connection, llm_fn) -> list[dict]:
    """
    質問に対してカテゴリツリーを上位からたどり、選ばれたカテゴリのパスを返す。
    """
    from switch_rag_objects import classify_question_by_llm
    from database import Category

    all_categories = Category.get_all_categories(conn)

    def find_children(parent_id: Optional[int]):
        return [cat for cat in all_categories if cat["parent_id"] == parent_id]

    path = []
    current_parent_id = 0

    while True:
        current_level = find_children(current_parent_id)
        if not current_level:
            break

        selector = {cat["name"]: cat.get("description") or "" for cat in current_level}
        scores = classify_question_by_llm(question, selector, llm_fn)
        selected_name = max(scores, key=scores.get)

        selected_category = next((cat for cat in current_level if cat["name"] == selected_name), None)
        if not selected_category:
            break

        path.append(selected_category)
        current_parent_id = selected_category["id"]

    return path



# Step 0: データベースを構築する
database.init_db(db_path, overwrite=True)
conn = database.db_connect(db_path)




# # Step 1 : PDFをMarkdownに変換する
# tebiki_pdf = pdf_dir / "tebiki.pdf"
# tebiki_md = markdown_dir / "tebiki.md"

# # PDFをMarkdownに変換する
# if not os.path.exists(tebiki_pdf):
#     print(f"PDFファイルが見つかりません: {tebiki_pdf}")

#     # PDFをMarkdownに変換する
#     document_utils.convert_document_to_markdown(tebiki_pdf, tebiki_md)

# # 出力されたMarkdownファイルの構造を確認する
# headings = collect_markdown_headings(Path(tebiki_md))
# for h in headings:
#     print(h)
# # process_markdown_structure(Path(tebiki_md))


# Step 2: Project オブジェクトの生成
new_project = database.Project(
    name="Sample",
    description="モジュールのサンプルコード",
    author="藤本悠",
    status="active",
    default_model_name="gemma3:4b",
    default_prompt_name="japanese_concise",
    default_embedding_name="bge-m3",
    notes="実装のテスト",
    dbcon=conn, 
    insert=True
)

# Step 3: Document の　Category オブジェクトの生成
cat_info = database.Category(
    name = "情報",
    description = "コンピューターに関連するカテゴリー",
    parent_id = 0,
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_civil = database.Category(
    name = "土木",
    description = "建設、建築、土木工事、測量に関するカテゴリー",
    parent_id = 0,
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_univ = database.Category(
    name = "大学",
    description = "大学の施設、教育、制度に関するカテゴリー",
    parent_id = 0,
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_std = database.Category(
    name = "学生",
    description = "大学生の学生生活に関するカテゴリー",
    parent_id = cat_univ.id,
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

# Step 3: Document オブジェクトの生成
rag_info = database.Document(
    project_id = new_project.id,
    category_id = cat_info.id,
    name ="cat_tools", 
    file_path = Path(markdown_dir / "cat_tools.md"),
    vectorstore_path = Path(vectorstore_dir / "cat_tools"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_univ = database.Document(
    project_id = new_project.id,
    category_id = cat_univ.id,
    name ="Proffesional_College_of_arts_and_tourism", 
    file_path = Path(markdown_dir / "Proffesional_College_of_arts_and_tourism.md"),
    vectorstore_path = Path(vectorstore_dir / "Proffesional_College_of_arts_and_tourism"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_civil = database.Document(
    project_id = new_project.id,
    category_id = cat_civil.id,
    name ="japan_catapillar", 
    file_path = Path(markdown_dir / "japan_catapillar.md"),
    vectorstore_path = Path(vectorstore_dir / "japan_catapillar"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_std = database.Document(
    project_id = new_project.id,
    category_id = cat_std.id,
    name ="students", 
    file_path = Path(markdown_dir / "students.md"),
    vectorstore_path = Path(vectorstore_dir / "students"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

# Step 4: Document からパラグラフを構築する
# new_project.build_vectorstore(entries, markdown_dir=markdown_dir, overwrite=False)




# カテゴリセレクタ
selector = database.get_category_selector(conn, parent_id=None)

def call_ollama(prompt):
    # 任意のLLM呼び出しコードに置き換えてください
    return ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": prompt}])["message"]["content"]

while True:
    question = input("質問を入力してください：\n＞ ")

    path = get_category_path_by_question(question, conn, call_ollama)

    if not path:
        print("カテゴリが見つかりませんでした。")
        continue

    print("➡️ 選ばれたカテゴリ階層:")
    for level, cat in enumerate(path, 1):
        print(f"Level {level}: {cat['name']}")

    final_cat = path[-1]
    answer = input(f"このカテゴリ「{final_cat['name']}」でよろしいですか？（はい/いいえ）\n＞ ")

    if answer.strip() in ["はい", "yes", "OK", "うん"]:
        print(f"✅ 「{final_cat['name']}」を使用してRAGを実行します。")
        # ここでベクトル検索・RAGチェーン準備へ進む（後述）
    else:
        print("もう一度、具体的な質問を入力してください。")


# # ベクトルストアを構築する






# # RAGによるチェーンを構築する
# category = HierarchicalRetrieverCategory(tagname="土木", parent_tag="root")
# session.prepare_chain(category=category)

# # 対話型の質問エージェントを起動する
# session.run_interactive(mode="rag")    # ← 従来通り
# # session.run_interactive(mode="llm")  # ← RAGを使わず直接生成
