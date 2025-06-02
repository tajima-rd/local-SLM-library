# coding: utf-8

# 外部ライブラリから読み込んだモジュール類
import re

from pathlib import Path
from typing import Optional
from langdetect import detect
from graphviz import Source # type: ignore

# 今回のプロジェクトのために開発した自作コード
from core import switch_rag_objects as sro
from core import document_utils as du
from core import database # type: ignore


# --- ディレクトリ設定 ---
# 現在のファイル（スクリプト）のパス
current_path = Path(__file__).resolve()
 
# 'core' ディレクトリを含む親ディレクトリを見つける
core_root = next(p for p in current_path.parents if p.name == "local-SLM-library")

# そこから目的のサブパスを定義
sample_dir = core_root / "sample"
pdf_dir = sample_dir / "pdf"


# データベースのパス
db_dir = core_root / "database"
db_path = db_dir / "database.db"

markdown_dir = db_dir / "markdown"
vectorstore_dir = db_dir / "vectorstore"


def save_tree_as_image(dot_string: str, output_path: str, format: str = "png") -> None:
    """
    DOT 形式のツリー文字列を画像に保存する。

    Args:
        dot_string: Graphviz DOT 記法の文字列
        output_path: 出力ファイルパス（拡張子は自動でつく）
        format: "png" または "svg"
    """
    if format not in ("png", "svg"):
        raise ValueError("format must be 'png' or 'svg'")

    s = Source(dot_string)
    s.render(output_path, format=format, cleanup=True)
    print(f"✅ Saved: {output_path}.{format}")

def tree_to_graphviz_dot(tree: list[dict], graph_name: str = "CategoryTree") -> str:
    """
    確率ツリーを Graphviz の DOT 形式に変換する。

    Args:
        tree: `build_probability_tree` による再帰的ツリー構造
        graph_name: グラフ名（任意）

    Returns:
        Graphviz DOT 形式の文字列
    """
    lines = [f'digraph {graph_name} {{', '  node [shape=box];']

    def add_node(node: dict, parent_ids: Optional[int] = None):
        node_id = f'node{node["id"]}'
        label = f'{node["name"]}\\n{node["score"]:.2f}'
        lines.append(f'  {node_id} [label="{label}"];')
        if parent_ids is not None:
            lines.append(f'  node{parent_ids} -> {node_id};')

        for child in node.get("children", []):
            add_node(child, node["id"])

    for root in tree:
        add_node(root)

    lines.append('}')
    return "\n".join(lines)


# Step 0: データベースを構築する
database.init_db(db_path, overwrite=True)
conn = database.db_connect(db_path)

# Step 1 : PDFをMarkdownに変換する
pdf_path = pdf_dir / "s12029-015-9724-1.pdf"
markdown_path = markdown_dir / "s12029-015-9724-1.md"
if not markdown_path.exists():
    # PDFをMarkdownに変換する
    du.convert_document_to_markdown(pdf_path, markdown_path)
    print(f"✅ PDFをMarkdownに変換しました: {markdown_path}")
else:
    print(f"❌ Markdownファイルは既に存在します: {markdown_path}")


# Step 2: Project オブジェクトの生成
new_project = database.Project(
    name="Sample",
    description="モジュールのサンプルコード",
    author="藤本悠",
    status="active",
    default_model_name="granite3.3:2b",
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
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_civil = database.Category(
    name = "土木",
    description = "建設、建築、土木工事、測量に関するカテゴリー",
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_medic = database.Category(
    name = "医療",
    description = "人の健康、病気、医療、看護、医薬品、医療機器に関するカテゴリー",
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_univ = database.Category(
    name = "大学",
    description = "大学の施設、教育、制度に関するカテゴリー",
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_std = database.Category(
    name = "学生",
    description = "大学生の学生生活に関するカテゴリー",
    parent_ids = [cat_univ.id],
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

rag_medic = database.Document(
    project_id = new_project.id,
    category_id = cat_medic.id,
    name ="Pancreatic_Cancer", 
    file_path = Path(markdown_dir / "s12029-015-9724-1.md"),
    vectorstore_dir = Path(vectorstore_dir / "Pancreatic_Cancer"),
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
selector = database.get_category_selector(conn, parent_ids=None)

while True:
    question = input("今回はどのようなテーマの質問がありますか？：\n＞ ")
    language = detect(question)
    print(f"🌐 質問言語判定: {language}")

    path = sro.get_category_path(question, conn, new_project.start_chat, language=language)
    # tree = sro.get_probability_tree(question, conn, new_project.start_chat, language=language)

    # gviz = tree_to_graphviz_dot(tree)
    # save_tree_as_image(gviz, "category_tree", format="png")

    if not path:
        print("カテゴリが見つかりませんでした。")
        continue

    print("➡️ 選ばれたカテゴリ階層:")
    for level, cat in enumerate(path, 1):
        print(f"Level {level}: {cat['name']}（id: {cat['id']}）")

    final_cat = path[-1]
    answer = input(f"このカテゴリ「{final_cat['name']}」でよろしいですか？（はい/いいえ）\n＞ ")

    if answer.strip().lower() in ["はい", "yes", "ok", "うん"]:
        print(f"✅ 「{final_cat['name']}」を使用してRAGを実行します。")

        # ✅ RAG対象ベクトルストアの言語を取得
        languages = database.Paragraph.get_languages_by_category_id(conn, final_cat["id"])
        print(f"📚 ベクトルストアで使われている言語: {languages}")

        # ✅ 日本語以外を含むなら翻訳を検討
        if "ja" not in languages and "en" in languages:
            print("🈶 RAGは英語ベース。質問文を翻訳します。")
            question_translated = sro.translate_to_japanese(question, new_project.start_chat)
        else:
            question_translated = question

        print(f"🎯 RAG用質問文:\n{question_translated}")


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
