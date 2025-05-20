# coding: utf-8

# 外部ライブラリから読み込んだモジュール類
import json, re
import sqlite3
from typing import Optional
from langdetect import detect, LangDetectException

def safe_detect_language(text: str, min_length: int = 20) -> str:
    text = text.strip()
    if not text or len(text) < min_length:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def translate_to_japanese(text: str, llm_fn) -> str:
    prompt = f"以下の英文を日本語に翻訳してください。\n\n{text}\n\n翻訳文のみを返してください。"
    output = llm_fn(prompt)
    return re.sub(r"^.*?翻訳\s*[:：]?\s*", "", output).strip()

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
    LLM応答から JSON ブロックを抽出し、不正なキーに対してダブルクオートを補う。
    """
    match = re.search(r'\{[\s\S]*?\}', text)
    if not match:
        raise ValueError("❌ JSON形式のブロックが見つかりませんでした")

    json_text = match.group(0)

    # ✅ ダブルクオートで囲まれていないキーを補完（例: C1: → "C1":）
    # 注意: 正規表現はJSON構造の簡易修正用であり、完璧な構文チェックではない
    json_text = re.sub(r'([{\s,])([A-Z][0-9]+)(\s*):', r'\1"\2"\3:', json_text)

    return json_text

def classify_question_by_llm(
    question: str,
    selector: dict[str, str],
    llm_fn,
    language: str  # ← ユーザー質問の言語（ja / en など）
) -> dict[str, float]:
    """
    カテゴリ名のバイアスを避けるため、匿名ラベルで分類させてから実カテゴリに復号する。
    英語の説明は LLM を使って日本語に翻訳する。
    """

    # --- 英語の説明文を日本語に翻訳する ---
    normalized_selector = {}
    for name, desc in selector.items():
        lang = safe_detect_language(desc)
        if lang == "en":
            desc = translate_to_japanese(desc, llm_fn)
            print(f"🌐 翻訳: {desc}")
        normalized_selector[name] = desc

    # --- ラベルマップの作成 ---
    label_map = {f"C{i+1}": (k, v) for i, (k, v) in enumerate(normalized_selector.items())}
    if "その他" not in normalized_selector:
        label_map["C999"] = ("その他", "上記のどれにも明確に分類されない場合")

    anonymized_prompt = "\n".join([f"- {label}: {desc}" for label, (_, desc) in label_map.items()])

    prompt = f"""
    次の質問が、以下のカテゴリ説明のどれに最も当てはまるかを、確率として評価してください。
    カテゴリ名は伏せています。説明だけを基に判断してください。

    {anonymized_prompt}

    判断基準:
    - 質問文の主題とカテゴリ説明との意味的な一致度を評価する
    - 質問文のキーワードが含まれるカテゴリを優先する
    - 各カテゴリの記述の網羅性・具体性を考慮する
    - 安全側（その他）に分類するのは最終手段とする

    質問文：
    {question}

    出力形式（JSONのみ）：
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

def get_category_path(
    question: str, 
    conn: sqlite3.Connection, 
    llm_fn,
    language: str
) -> list[dict]:
    """
    質問に対してカテゴリツリーを上位からリーフまでたどり、最終的にリーフカテゴリのパス（id含む）を返す。
    """
    from switch_rag_objects import classify_question_by_llm
    from database import Category

    all_categories = Category.get_all_categories(conn)

    def find_children(parent_id: Optional[int]):
        return [cat for cat in all_categories if cat["parent_id"] == parent_id]

    def find_path_recursive(current_parent_id: Optional[int], current_path: list[dict]) -> list[dict]:
        children = find_children(current_parent_id)
        if not children:
            return current_path  # リーフに到達

        # 各子カテゴリに対する説明を使ってスコアを付ける
        selector = {cat["name"]: cat.get("description") or "" for cat in children}
        scores = classify_question_by_llm(question, selector, llm_fn, language)

        # スコア最大のカテゴリを選択
        selected_name = max(scores, key=scores.get)
        selected_cat = next((cat for cat in children if cat["name"] == selected_name), None)

        if not selected_cat:
            return current_path  # 異常ケース

        # 選ばれたカテゴリをパスに追加
        current_path.append({
            "id": selected_cat["id"],
            "name": selected_cat["name"],
            "description": selected_cat.get("description"),
            "parent_id": selected_cat["parent_id"]
        })

        # 再帰的にその子カテゴリを探索
        return find_path_recursive(selected_cat["id"], current_path)

    # ルート（parent_id=Noneまたは0）から探索開始
    return find_path_recursive(0, [])

def get_probability_tree(
    question: str,
    conn: sqlite3.Connection,
    llm_fn,
    language:str,
    parent_id: Optional[int] = 0,
    threshold: float = 0.00
) -> list[dict]:
    """
    質問に対するカテゴリ確率ツリーを構築する。

    Args:
        question: 質問文
        conn: SQLite 接続
        llm_fn: LLM 推論関数
        parent_id: 現在の階層の親ID
        threshold: 分岐を記録する確率の最小値（ノイズ削減のため）

    Returns:
        各カテゴリノードを含むリスト。各ノードは `children` を再帰的に持つ。
    """
    from switch_rag_objects import classify_question_by_llm
    from database import Category

    all_categories = Category.get_all_categories(conn)
    current_level = [cat for cat in all_categories if cat["parent_id"] == parent_id]

    if not current_level:
        return []

    selector = {cat["name"]: cat.get("description") or "" for cat in current_level}
    scores = classify_question_by_llm(question, selector, llm_fn, language)

    result = []
    for cat in current_level:
        score = scores.get(cat["name"], 0.0)
        if score >= threshold:
            subtree = get_probability_tree(
                question=question,
                conn=conn,
                llm_fn=llm_fn,
                language=language,
                parent_id=cat["id"],
                threshold=threshold
            )
            result.append({
                "id": cat["id"],
                "name": cat["name"],
                "description": cat.get("description"),
                "parent_id": cat["parent_id"],
                "score": score,
                "children": subtree
            })

    return result