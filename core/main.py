from pathlib import Path
from RAGSession import RAGSession  # type: ignore
from retriever_utils import HierarchicalRetrieverCategory
import database # type: ignore


import json, ollama
import re
from typing import Optional


def build_classification_response(
    probabilities: dict[str, float],
    threshold: float = 0.6
) -> tuple[str, Optional[str]]:
    """
    分類結果に基づく確認・誘導メッセージを生成する。

    Returns:
        message: ユーザーへのメッセージ
        suggested_category: 同意が得られた場合に使用するカテゴリ名（再質問時は None）
    """
    best_category = max(probabilities, key=probabilities.get)
    best_prob = probabilities[best_category]

    if best_prob < threshold:
        category_list = ", ".join(probabilities.keys())
        message = (
            "その質問は複数のカテゴリーにまたがる可能性があります。\n"
            "もう少し具体的に聞いてください。\n"
            f"現時点で扱えるカテゴリーは次のとおりです：{category_list}"
        )
        return message, None
    else:
        message = (
            f"この質問は「{best_category}」のカテゴリーに分類されました。\n"
            "この分類で問題ありませんか？（はい／いいえ でお答えください）"
        )
        return message, best_category

def extract_json_block(text: str) -> str:
    """
    テキスト中から最初に出現するJSON形式のブロック（{...}）を抽出する
    """
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        return match.group(0)
    raise ValueError("JSONブロックが見つかりませんでした。")

def classify_question_by_llm(question: str, selector: dict[str, str], llm_fn) -> dict[str, float]:
    """
    カテゴリ名のバイアスを避けるため、匿名ラベルで分類させてから実カテゴリに復号する。
    """
    # --- ラベルマップの作成 ---
    label_map = {f"C{i+1}": (k, v) for i, (k, v) in enumerate(selector.items())}
    if "その他" not in selector:
        label_map["C999"] = ("その他", "上記のどれにも明確に分類されない場合")

    # --- プロンプト生成 ---
    anonymized_prompt = "\n".join([f"- {label}: {desc}" for label, (_, desc) in label_map.items()])

    prompt = f"""
        次の質問が、以下のカテゴリ説明のどれに最も当てはまるかを、確率として評価してください。
        カテゴリ名は伏せています。説明だけを基に判断してください。

        {anonymized_prompt}

        質問文：
        {question}

        出力形式（JSON）：
        {json.dumps({label: 0.0 for label in label_map.keys()}, ensure_ascii=False, indent=2)}
    """.strip()

    response = llm_fn(prompt)
    print(response)

    try:
        json_text = extract_json_block(response)
        result = json.loads(json_text)
        total = sum(result.values())
        if total > 0:
            normalized = {label_map[k][0]: v / total for k, v in result.items() if k in label_map}
        else:
            normalized = {label_map[k][0]: 0.0 for k in result if k in label_map}
        return normalized
    except Exception as e:
        raise ValueError(f"LLMからの出力を解析できませんでした:\n{response}") from e


# --- ディレクトリ設定 ---
# 現在のファイル（スクリプト）のパス
current_path = Path(__file__).resolve()
 
# 'core' ディレクトリを含む親ディレクトリを見つける
core_root = next(p for p in current_path.parents if p.name == "core")

# そこから目的のサブパスを定義
base_dir = core_root / "sample"
markdown_dir = base_dir / "markdown"
vectorstore_dir = base_dir / "vectorstore"
db_path = base_dir / "database.db"

# Step 0: データベースを構築する
database.init_db(db_path, overwrite=True)
conn = database.db_connect(db_path)

# Step 1: Project オブジェクトの生成
new_project = database.Project(
    name="Sample",
    description="モジュールのサンプルコード",
    author="藤本悠",
    status="active",
    default_prompt="japanese_concise",
    default_embedding="bge-m3",
    notes="実装のテスト",
    dbcon=conn, 
    insert=True
)

# Step 2: Category オブジェクトの生成
cat_info = database.Category(
    name = "情報",
    description = "コンピューターに関連するカテゴリー",
    parent_tag = "root",
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_civil = database.Category(
    name = "土木",
    description = "建設、建築、土木工事、測量に関するカテゴリー",
    parent_tag = "root",
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_univ = database.Category(
    name = "大学",
    description = "大学の施設、教育、制度に関するカテゴリー",
    parent_tag = "root",
    type_code = "hier",
    sort_order = 0,
    dbcon=conn, 
    insert=True
)

cat_std = database.Category(
    name = "学生",
    description = "大学生の学生生活に関するカテゴリー",
    parent_tag = "大学",
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
    vectorstore_path = str(vectorstore_dir / "cat_tools"),
    embedding_model = "bge-m3",
    dbcon=conn, 
    insert=True
)

rag_univ = database.Document(
    project_id = new_project.id,
    category_id = cat_univ.id,
    name ="Proffesional_College_of_arts_and_tourism", 
    file_path = str(markdown_dir / "Proffesional_College_of_arts_and_tourism.md"),
    vectorstore_path = str(vectorstore_dir / "Proffesional_College_of_arts_and_tourism"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_civil = database.Document(
    project_id = new_project.id,
    category_id = cat_civil.id,
    name ="japan_catapillar", 
    file_path = str(markdown_dir / "japan_catapillar.md"),
    vectorstore_path = str(vectorstore_dir / "japan_catapillar"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

rag_std = database.Document(
    project_id = new_project.id,
    category_id = cat_std.id,
    name ="students", 
    file_path = str(markdown_dir / "students.md"),
    vectorstore_path = str(vectorstore_dir / "students"),
    embedding_model = new_project.default_embedding,
    dbcon=conn, 
    insert=True
)

# カテゴリセレクタ
selector = database.get_category_selector(conn, parent_tag="root")
database.db_close(conn)

def call_ollama(prompt):
    # 任意のLLM呼び出しコードに置き換えてください
    return ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": prompt}])["message"]["content"]

while True:
        question = input("質問を入力してください：\n＞ ")

        probabilities = classify_question_by_llm(question, selector, call_ollama)
        print(probabilities)

        message, suggested = build_classification_response(probabilities, threshold=0.6)
        print(message)

        if suggested is not None:
            answer = input("＞ ")
            if answer.strip() in ["はい", "yes", "OK", "うん"]:
                print(f"では「{suggested}」カテゴリーとして処理を続けます。")
                # 本処理へ
            else:
                print("もう一度、具体的な質問を入力してください。")
        else:
            print("再度、質問をどうぞ。")




# # セッション作成
# session = RAGSession(
#     vectorstore_dir=vectorstore_dir,
#     model_name="gemma3:4b",
#     default_template="japanese_concise",
#     embedding_name="bge-m3"
# )

# # ベクトルストアを構築する
# session.build_vectorstore(entries, markdown_dir=markdown_dir, overwrite=False)

# # RAGによるチェーンを構築する
# category = HierarchicalRetrieverCategory(tagname="土木", parent_tag="root")
# session.prepare_chain(category=category)

# # 対話型の質問エージェントを起動する
# session.run_interactive(mode="rag")    # ← 従来通り
# # session.run_interactive(mode="llm")  # ← RAGを使わず直接生成
