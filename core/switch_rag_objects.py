# coding: utf-8

# 外部ライブラリから読み込んだモジュール類
import json, re
import sqlite3
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

def get_category_path(
        question: str, 
        conn: sqlite3.Connection, 
        llm_fn
) -> list[dict]:
    """
    質問に対してカテゴリツリーを上位からたどり、選ばれたカテゴリのパス（id含む）を返す。
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

        path.append({
            "id": selected_category["id"],
            "name": selected_category["name"],
            "description": selected_category.get("description"),
            "parent_id": selected_category["parent_id"]
        })
        current_parent_id = selected_category["id"]

    return path