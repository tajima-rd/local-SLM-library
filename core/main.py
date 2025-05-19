# coding: utf-8

# 外部ライブラリから読み込んだモジュール類
from pathlib import Path
import database # type: ignore
from typing import Optional, List, Any, Tuple

# 今回のプロジェクトのために開発した自作コード
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

# Step 0: データベースを構築する
database.init_db(db_path, overwrite=True)
conn = database.db_connect(db_path)

# Step 1 : PDFをMarkdownに変換する


# Step 2: Project オブジェクトの生成
new_project = database.Project(
    name="Sample",
    description="モジュールのサンプルコード",
    author="藤本悠",
    status="active",
    default_model_name="granite3.3:8b",
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


# Step 4: Document オブジェクトの生成
rag_info = database.Document(
    project_id = new_project.id,
    category_id = cat_info.id,
    name ="cat_tools", 
    file_path = Path(markdown_dir / "cat_tools.md"),
    vectorstore_dir = Path(vectorstore_dir / "cat_tools"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_univ = database.Document(
    project_id = new_project.id,
    category_id = cat_univ.id,
    name ="Proffesional_College_of_arts_and_tourism", 
    file_path = Path(markdown_dir / "Proffesional_College_of_arts_and_tourism.md"),
    vectorstore_dir = Path(vectorstore_dir / "Proffesional_College_of_arts_and_tourism"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_civil = database.Document(
    project_id = new_project.id,
    category_id = cat_civil.id,
    name ="japan_catapillar", 
    file_path = Path(markdown_dir / "japan_catapillar.md"),
    vectorstore_dir = Path(vectorstore_dir / "japan_catapillar"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_std = database.Document(
    project_id = new_project.id,
    category_id = cat_std.id,
    name ="students", 
    file_path = Path(markdown_dir / "students.md"),
    vectorstore_dir = Path(vectorstore_dir / "students"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

# カテゴリセレクタの取得（未使用なら削除可）
selector = database.get_category_selector(conn, parent_id=None)

while True:
    question = input("今回はどのようなテーマの質問がありますか？：\n＞ ")

    # ✅ llm_fn には new_project.start_chat という「関数オブジェクト」を渡す
    path = sro.get_category_path(question, conn, new_project.start_chat)

    if not path:
        print("カテゴリが見つかりませんでした。")
        continue

    print("➡️ 選ばれたカテゴリ階層:")
    for level, cat in enumerate(path, 1):
        print(f"Level {level}: {cat['name']}（id: {cat['id']}）")

    final_cat = path[-1]
    answer = input(f"このカテゴリ「{final_cat['name']}」でよろしいですか？（はい/いいえ）\n＞ ")

    if answer.strip() in ["はい", "yes", "OK", "うん"]:
        print(f"✅ 「{final_cat['name']}」を使用してRAGを実行します。")

        # ✅ ベクトルストアのパスを取得（最初の1件を使う）
        paths = database.Paragraph.get_vectorstore_by_category_id(conn, final_cat['id'])
        if not paths:
            print("❌ ベクトルストアが見つかりませんでした。")
        else:
            from chain_factory import prepare_chain_from_path
            from retriever_utils import FlatRetrieverCategory

            # ✅ カテゴリを FlatRetrieverCategory に変換
            category = FlatRetrieverCategory(tagname=final_cat["name"])

            # ✅ チェーン構築
            try:
                rag_chain = prepare_chain_from_path(
                    llm=new_project.rag_session.llm,
                    faiss_paths=paths,
                    chain_type="conversational",  # または retrievalqa, stuff など
                    k=5,  # top-k 検索数
                    prompt_template=new_project.rag_session.prompt_template
                )

                # ✅ ユーザー質問を投げる
                user_query = input("質問を入力してください：\n＞ ")
                result = rag_chain.invoke({"input": user_query})
                print("🧠 応答:", result["answer"] if isinstance(result, dict) else result)

            except FileNotFoundError as e:
                print("❌ ベクトルストアの読み込みに失敗しました:", e)

        break  # ループ終了（必要なら）
    else:
        print("もう一度、具体的な質問を入力してください。")
