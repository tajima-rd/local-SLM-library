# coding: utf-8

# 外部ライブラリから読み込んだモジュール類
import json, re
import sqlite3
from typing import Optional, List, Dict, Any, Tuple # 型ヒントをインポート
from langdetect import detect, LangDetectException # detect, LangDetectException をインポート

# 今回のプロジェクトのために開発した自作コード
from objects import database # 相対インポートに変更。database モジュール全体が必要。

def safe_detect_language(text: str, min_length: int = 20) -> str:
    text = text.strip()
    if not text or len(text) < min_length:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        # print(f"⚠️ 言語検出エラー for text snippet: {text[:50]}...") # デバッグ用
        return "unknown"
    except Exception as e:
        # print(f"⚠️ 予期せぬ言語検出エラー: {e} for text snippet: {text[:50]}...") # デバッグ用
        return "unknown"


# 注: llm_obj は実際には Ollama の chat 関数のようなものを受け取る想定か、
# または LangChain の LLM オブジェクトの invoke/generate メソッドのようなものかによる。
# ここでは LangChain LLM オブジェクトを使う想定で llm_obj.invoke() を使用。
def translate_to_japanese(text: str, llm_obj: Any) -> str:
    """
    LLM オブジェクトを使用して英文を日本語に翻訳する。
    llm_obj は invoke メソッドを持つ LangChain LLM オブジェクトなどを想定。
    """
    if not text.strip():
         return "" # 空白のみの場合は空文字列を返す

    prompt = f"以下の英文を日本語に翻訳してください。\n\n{text}\n\n翻訳文のみを返してください。"
    
    try:
        # LangChain LLM オブジェクトの invoke メソッドを呼び出す想定
        # OllamaEmbeddings ではなく、チャットモデルやテキスト生成モデルが必要
        # invoke の返り値は LangChain の Message オブジェクトや文字列などモデルによるので注意
        llm_response_obj = llm_obj.invoke(prompt)
        output_text = llm_response_obj.content if hasattr(llm_response_obj, 'content') else str(llm_response_obj)

        # 余分なテキストを除去
        # re.sub(r"^.*?翻訳\s*[:：]?\s*", "", output_text).strip() は翻訳結果がこの形式にならない場合に問題
        # シンプルにstrip()だけにするか、プロンプトで出力形式を厳密に指定する
        return output_text.strip()
    except Exception as e:
        print(f"❌ 翻訳中にエラーが発生しました: {e}")
        # traceback.print_exc() # 翻訳エラーは頻繁に出る可能性があるのでトレースバックはコメントアウト
        return f"翻訳エラー: {e}" # エラーメッセージを返すか、元のテキストを返すか


def build_classification_response(
    probabilities: dict[str, float],
    threshold: float = 0.6
) -> tuple[str, Optional[str]]:
    """
    分類結果に基づく確認・誘導メッセージを生成する。
    この関数は get_probability_tree の結果（Dictのリスト形式）ではなく、
    classify_question_by_llm の結果（{カテゴリ名: 確率} Dict）を直接受け取ることを想定している古いロジック。
    現在は get_probability_tree の結果からパスを選ぶ方が一般的。
    この関数が現在も必要か確認が必要。必要でなければ削除を検討。
    sample_01_rag_construction.pyでは使われていないため、現状はそのまま残すが注意。

    Returns:
        message: ユーザーへのメッセージ
        suggested_category_name: 同意が得られた場合に使用するカテゴリ名（再質問時は None）
    """
    if not probabilities:
         return "分類可能なカテゴリが見つかりませんでした。", None
         
    # スコアの降順でソート
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    
    best_category_name = sorted_probs[0][0]
    best_prob = sorted_probs[0][1]

    if best_prob < threshold:
        # スコアが閾値未満の場合、選択肢を提示し具体化を促す
        # 確率の高い順に数個表示するのが良い
        top_n = 5 # 例: 上位5個を表示
        # ensure sorted_probs has enough items
        suggested_categories = [f"「{cat_name}」（確度: {prob:.2f}）" for cat_name, prob in sorted_probs[:min(top_n, len(sorted_probs))]]
        category_list_str = "、".join(suggested_categories)

        message = (
            "その質問に対する確度の高いカテゴリが複数あるか、どれも十分な確度を持ちません。\n"
            "もう少し具体的に質問内容を言い換えていただけますでしょうか？\n"
            f"現時点で関連性の高いと思われるカテゴリー（確度順）:\n{category_list_str}"
        )
        return message, None
    else:
        # スコアが閾値以上の場合、最尤カテゴリを提案
        message = (
            f"この質問は「{best_category_name}」のカテゴリーに分類されました (確度: {best_prob:.2f})。\n"
            "この分類で問題ありませんか？（はい／いいえ でお答えください）"
        )
        # 提案するカテゴリとしては、名前を返す
        return message, best_category_name


def extract_json_block(text: str) -> str:
    """
    LLM応答から JSON ブロックを抽出し、不正なキーに対してダブルクオートを補う。
    より頑健な抽出と修正を試みる。
    """
    # JSONlikelな部分を探す（先頭または行頭の{から末尾または行末の}まで）
    # ```json ... ``` のようなコードブロック内も考慮
    code_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        json_text = code_block_match.group(1)
    else:
        # コードブロックが見つからない場合は単純な {} 検索
        match = re.search(r'\{[\s\S]*?\}', text)
        if not match:
             # print(f"⚠️ JSONブロックが見つかりませんでした。元のテキスト:\n{text}") # デバッグ用
             raise ValueError("❌ JSON形式のブロックが見つかりませんでした")
        json_text = match.group(0) # 抽出した JSON 候補文字列


    # 簡易的な修正:
    # - 行コメント // ... を削除
    # - 末尾カンマの削除（オブジェクトや配列の最後の要素の後ろのカンマ）
    # - ダブルクオートで囲まれていないキー（英数字+_のみを想定）にダブルクオートを追加
    #   例: key: "value" -> "key": "value"
    #   これは非常に難しい問題で、完璧な正規表現はない。あくまでLLMの癖に対する簡易対応。
    
    # 行コメント // を削除
    json_text = re.sub(r'//.*', '', json_text)
    
    # 末尾カンマを削除 (オブジェクト {} または配列 [] の直前のカンマ)
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

    # キーの修正 (例: { key: value } -> { "key": value })
    # 正規表現 r'([{,]\s*)(\w+)(\s*:)' は、{ または , の後にある単語(\w+)とそれに続くコロンを検出
    # これを $1"$2"$3 に置換して単語をダブルクオートで囲む
    # ただし、既にダブルクオートで囲まれているキーもマッチする可能性があるので注意が必要
    # より安全な正規表現: [{,]\s*(\w+)(?:\s*): は非キャプチャグループ(?:\s*)でコロンの前の空白をマッチさせる
    # そして、後ろにダブルクオートがないことを(?!\s*")でチェック
    # r'([{,]\s*)((?:"[^"]*"|\w+))(\s*:)' # ダブルクオートされたキーまたは単語をマッチ
    # r'([{,]\s*)(\w+)(\s*:)' # ダブルクオートされていない単語キーのみにマッチ

    # LLMの出力傾向に合わせて調整が必要。ここではシンプルに、{ または , の後に来て、コロンで終わる単語をキーとみなす。
    # ただし、既にダブルクオートで囲まれていない単語で、かつコロンが続くパターンに限定
    # r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)' # 英数字とアンダースコアで始まる単語キー
    json_text = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', json_text)


    # デバッグ用
    # print("DEBUG: Extracted and potentially fixed JSON text:")
    # print(json_text)

    return json_text

# self-import を削除
# from switch_rag_objects import classify_question_by_llm # 削除済み

# database も相対インポートで既にインポート済み
# from database import Category # 不要、既に上でインポート済み

def classify_question_by_llm(
    question: str,
    selector: dict[str, str], # {カテゴリ名: 説明}
    llm_obj: Any, # LangChain LLM オブジェクトなどを想定
    language: str  # ← ユーザー質問の言語（ja / en など）
) -> dict[str, float]:
    """
    カテゴリ名のバイアスを避けるため、匿名ラベルで分類させてから実カテゴリに復号する。
    英語の説明は LLM を使って日本語に翻訳する（LLM_objをtranslate_to_japaneseに渡す）。
    """
    if not selector:
        # print("⚠️ classify_question_by_llm に空のセレクターが渡されました。") # これはツリー末端で発生するので毎回出すとノイズになる
        return {} # 空のセレクターの場合は空辞書を返す


    # --- 英語の説明文を日本語に翻訳する（必要なら）---
    # LLMが多言語対応していれば翻訳は不要かもしれないが、ここでは翻訳するロジックを残す
    # 翻訳には同じ llm_obj を使う想定
    translated_selector_desc = {}
    for name, desc in selector.items():
        if desc and isinstance(desc, str): # 説明が空でない文字列の場合のみ翻訳を試みる
            try:
                # 説明文の言語を検出
                desc_lang = safe_detect_language(desc)
                # ユーザーの質問言語と異なる場合、または未知でない場合に翻訳
                # ここではシンプルに、元の言語が英語で、ユーザーの質問言語が日本語の場合に翻訳
                if desc_lang == "en" and language == "ja":
                   # print(f"🌐 説明翻訳 (en->ja): '{desc}'") # デバッグ用
                   translated_desc = translate_to_japanese(desc, llm_obj)
                   # print(f"  -> '{translated_desc}'") # デバッグ用
                   translated_selector_desc[name] = translated_desc
                else:
                    translated_selector_desc[name] = desc # 翻訳しないか、日本語以外から日本語への翻訳など他のケース
            except Exception as e:
                # print(f"⚠️ 説明文 '{desc}' の言語検出または翻訳中にエラー: {e}") # デバッグ用
                translated_selector_desc[name] = desc # エラー時は元の説明を使用
        else:
             translated_selector_desc[name] = "" # 説明が None や空文字列なら空文字列


    # --- ラベルマップの作成 ---
    # {匿名ラベル: (カテゴリ名, 翻訳/元の説明)} のマップ
    label_map = {f"C{i+1}": (name, desc) for i, (name, desc) in enumerate(translated_selector_desc.items())}
    
    # その他カテゴリの追加
    # 匿名ラベルC999を「その他」に対応させる
    # その他カテゴリは常にリストに含める (LLMに「その他」という選択肢を与えるため)
    other_category_name = "その他"
    other_category_desc = "上記のカテゴリのどれにも明確に分類されない場合"
    
    # もしセレクターに既に「その他」があればそれを利用 (ただし匿名ラベルマップの最後に配置)
    # 既存のセレクターにその他カテゴリがある場合、その説明を使う
    existing_other_desc = selector.get(other_category_name, other_category_desc)
    # ただし、匿名ラベルマップは0から始まる連番で生成されるので、その他は最後に別途追加する方が安全
    label_map["C999"] = (other_category_name, existing_other_desc)


    anonymized_prompt_parts = [f"- {label}: {desc}" for label, (_, desc) in label_map.items()]
    anonymized_prompt_str = "\n".join(anonymized_prompt_parts)

    # LLMへの指示プロンプトを構築
    # プロンプトも llm_obj が対応する言語に合わせる必要があるかもしれない
    # ここでは日本語で固定
    prompt = f"""
    次の質問が、以下のカテゴリ説明のどれに最も当てはまるかを、確率として評価してください。
    カテゴリ名は匿名ラベル(例: C1, C2)で伏せています。説明だけを基に判断してください。

    カテゴリ説明リスト:
    {anonymized_prompt_str}

    評価基準:
    - 質問文の主題とカテゴリ説明との意味的な一致度を評価する
    - 質問文のキーワードが含まれるカテゴリを優先する
    - 各カテゴリの記述の網羅性・具体性を考慮する
    - 安全側（ラベル C999）に分類するのは最終手段とする

    回答形式:
    質問文に対する各匿名カテゴリラベルの確率を0.0から1.0の浮動小数点数で示し、合計が1.0に近くなるようにしてください。
    結果をJSON形式で出力してください。JSON以外の文字列は含めないでください。出力はJSONオブジェクトのみとし、コードブロック(```json)やその他の説明テキストを含めないでください。
    例:
    {{
      "C1": 0.85,
      "C2": 0.10,
      "C999": 0.05
    }}

    質問文：
    {question}
    """.strip()

    # LLM に推論を依頼
    # print("🧠 カテゴリ推論LLM呼び出し中...") # デバッグ用
    response = ""
    try:
         # LangChain LLM オブジェクトの invoke メソッドを呼び出す想定
         # response は LangChain の Message オブジェクトや文字列などモデルによる
         # timeout も設定可能か検討
         llm_response_obj = llm_obj.invoke(prompt) # add config={"max_retries": 3} など？
         response = llm_response_obj.content if hasattr(llm_response_obj, 'content') else str(llm_response_obj)
         # print(f"DEBUG: LLM Response:\n{response}") # デバッグ用応答全体出力

    except Exception as e:
        print(f"❌ LLM呼び出し中にエラーが発生しました: {e}")
        # traceback.print_exc() # LLM呼び出しエラーは頻繁に出る可能性があるのでトレースバックはコメントアウト
        # LLM呼び出し失敗時は空辞書を返す
        return {}


    # LLM応答から JSON を抽出・解析
    result = {}
    try:
        json_text = extract_json_block(response)
        result = json.loads(json_text)

        # 結果のバリデーションと正規化のための前処理
        if not isinstance(result, dict):
             # print(f"⚠️ LLM応答はJSONオブジェクトではありません。元の応答:\n{response}") # デバッグ用
             return {} # 解析失敗

        valid_result = {}
        for key, value in result.items():
             # 匿名ラベルが label_map に存在し、値が数値であれば有効とみなす
             if key in label_map:
                  try:
                       valid_result[key] = float(value)
                  except (ValueError, TypeError):
                       # print(f"⚠️ 結果の確率が数値ではありません ({key}: {value})。無視します。") # デバッグ用
                       pass # 数値でない場合は無視

        # 結果のバリデーション (全ての匿名ラベルが結果に含まれているかなど)
        # LLMが一部のラベルしか返さないこともあるため、必須チェックは難しい

        # 確率の正規化 (合計が1.0になるように)
        total = sum(valid_result.values())
        
        normalized_probs: Dict[str, float] = {}
        if total > 0:
            # 有効なラベルのみを正規化
            normalized_probs = {label_map[k][0]: v / total for k, v in valid_result.items()}
            
            # 正規化後の合計が1.0に近くなるように微調整
            # sum_normalized = sum(normalized_probs.values())
            # if abs(sum_normalized - 1.0) > 1e-6:
            #      print(f"⚠️ 正規化後の合計確率が1.0になりませんでした ({sum_normalized:.2f})。") # デバッグ用

        else:
            # 合計が0の場合、全ての確率を0とする
            # もし「その他」カテゴリが label_map に含まれていれば、「その他」に1.0を割り当てるなどのフォールバックも有効
             print(f"⚠️ LLM推論の合計確率が0でした。'{other_category_name}' に確率 1.0 を割り当てます。")
             # label_map から 'その他' のカテゴリ名を探す
             other_cat_name = next((name for label, (name, desc) in label_map.items() if label == "C999"), None)
             if other_cat_name:
                  normalized_probs[other_cat_name] = 1.0
             # その他のラベルは全て0
             for label, (name, desc) in label_map.items():
                  if name != other_cat_name:
                       normalized_probs[name] = 0.0


        return normalized_probs # {カテゴリ名: 正規化された確率} の辞書を返す

    except (json.JSONDecodeError, ValueError) as e:
        # print(f"❌ LLMからのJSON出力を解析できませんでした: {e}") # デバッグ用
        # print(f"元のLLM応答:\n{response}") # デバッグ用
        # traceback.print_exc() # JSON解析エラーは頻繁に出る可能性があるのでトレースバックはコメントアウト
        # 解析失敗時は空辞書やエラーを示す値を返す
        return {} # または raise ValueError("JSON解析失敗")
    except Exception as e:
        print(f"❌ LLM応答の処理中に予期せぬエラーが発生しました: {e}")
        # traceback.print_exc() # デバッグ用
        return {} # または raise


# self-import を削除
# from switch_rag_objects import classify_question_by_llm # 削除済み
# from database import Category # 不要、既に上でインポート済み

def get_category_path(
    question: str,
    conn: sqlite3.Connection,
    llm_obj: Any, # LangChain LLM オブジェクトなどを想定
    language: str
) -> List[Dict[str, Any]]: # 戻り値の型ヒントを Dict のリストに修正
    """
    質問に対してカテゴリツリーを上位からリーフまでたどり、最終的にリーフカテゴリまでのパス（id含む）を返す。
    パス上の各ノードでLLMによる分類推論を行う（貪欲法）。
    注意: この関数は get_probability_tree とロジックが重複しており、非推奨となる可能性がある。
    代わりに get_probability_tree の結果からパスを抽出するロジックを使用することを推奨。
    """
    # database モジュールは既に . import database as db とインポートされていることを前提とする

    all_categories = database.Category.get_all_categories(conn)
    if not all_categories:
        print("⚠️ DBにカテゴリがありません。パス探索できません。")
        return []

    def find_children(parent_cat_id: Optional[int]) -> List[database.Category]:
        """指定された親IDを持つ直接の子カテゴリのリストを返す"""
        children = []
        # 全カテゴリを走査し、その parent_ids リストに指定された親IDが含まれるものを探す
        for cat in all_categories:
             # cat.parent_ids はリスト
             # parent_cat_id が None の場合はルートカテゴリ（親がいないカテゴリ）を探す
             if parent_cat_id is None:
                 # 親IDリストが空のカテゴリがルート
                 if not cat.parent_ids and cat.id is not None: # IDがNoneでないかもチェック
                      children.append(cat)
             elif cat.id is not None and parent_cat_id is not None and parent_cat_id in cat.parent_ids: # 親IDリストに指定IDが含まれているかチェック
                 children.append(cat)
        
        # sort_order でソートする (必要であれば)
        children.sort(key=lambda c: c.sort_order)
        return children

    # 再帰関数でパスを探索し、Categoryオブジェクトのリストとして返す
    def find_path_recursive(current_parent_id: Optional[int], current_path_objs: List[database.Category]) -> List[database.Category]:
        """
        再帰的にパスを探索し、Categoryオブジェクトのリストとして返す。
        """
        children_cats = find_children(current_parent_id)

        if not children_cats:
            # リーフに到達、または子カテゴリが見つからない
            # ここでパス上の最後のカテゴリがリーフかどうか判断できる
            return current_path_objs # Categoryオブジェクトのリストを返す

        # 各子カテゴリに対する説明を使ってスコアを付ける
        # Categoryオブジェクトからセレクター辞書を作成
        selector = {cat.name: cat.description or cat.name for cat in children_cats}

        if not selector:
             # 子カテゴリはいるが有効なセレクター情報がない場合 (例: 全て description が None/空)
             return current_path_objs # 現在のパスで停止

        # LLM に推論を依頼し、スコアを取得
        scores = classify_question_by_llm(question, selector, llm_obj, language)

        if not scores:
             # LLM推論が失敗した場合や、空辞書が返された場合など
             print(f"⚠️ parent_id={current_parent_id} の階層でカテゴリ推論スコアが取得できませんでした。パス探索を中断します。")
             return current_path_objs # 現在のパスで停止


        # スコア最大のカテゴリを選択 (スコアが0の場合は選ばないなどの閾値判断も必要かも)
        # スコアが最も高いカテゴリ名を取得
        # max() 関数は空のiterableに対してValueErrorを発生させるため、scoresが空でないことを確認
        if not scores:
            print(f"⚠️ parent_id={current_parent_id} の階層で推論結果が空でした。パス探索を中断します。")
            return current_path_objs # scoresが空の場合は停止

        # scores辞書が空でない場合、最大スコアのキーを取得
        selected_name = max(scores, key=scores.get)
        selected_score = scores.get(selected_name, 0.0) # 念のためスコアも取得
        
        # スコアが低い場合はパス探索を打ち切る判断も有効
        selection_threshold = 0.1 # 例: 子の中で最も高いスコアがこの閾値未満なら停止
        if selected_score < selection_threshold:
             # print(f"⚠️ 子カテゴリの最高スコア ({selected_score:.2f}) が低いため、パス探索を打ち切ります。") # デバッグ用
             return current_path_objs # 現在のパスで停止


        # 選ばれた名前から対応する Category オブジェクトを探す
        selected_cat = next((cat for cat in children_cats if cat.name == selected_name), None)

        if not selected_cat or selected_cat.id is None:
            # 選ばれた名前のカテゴリがリストに見つからない、またはIDがない（異常ケース）
            print(f"❌ 選ばれたカテゴリ '{selected_name}' がCategoryオブジェクトリストに見つからないかIDがありません (parent_id={current_parent_id})。パス探索を中断します。")
            return current_path_objs  # 異常ケース、現在のパスで停止

        # 選ばれたカテゴリをパスに追加
        # current_path_objs は Category オブジェクトのリスト
        current_path_objs.append(selected_cat)

        # 再帰的にその子カテゴリを探索
        return find_path_recursive(selected_cat.id, current_path_objs)

    # --- パス探索開始 ---
    # ルートカテゴリ（parent_ids が空リスト）から探索開始
    # まずルートカテゴリを取得
    root_categories = find_children(None) # parent_id=None で親がいないカテゴリを探す

    # 複数のルートカテゴリがある場合、質問文でどのルートが最も適切か判断する必要がある
    if len(root_categories) > 1:
         print(f"ℹ️ 複数のルートカテゴリが見つかりました ({len(root_categories)} 件)。質問に基づいて最適なルートを選択します。")
         selector = {cat.name: cat.description or cat.name for cat in root_categories}
         
         if not selector:
             print("❌ ルートカテゴリに有効なセレクター情報がないため、パス探索できません。")
             return []

         scores = classify_question_by_llm(question, selector, llm_obj, language)

         if not scores:
              print("❌ ルートカテゴリの推論スコアが取得できませんでした。パス探索できません。")
              return []

         # スコア最大のルートカテゴリを選択
         # scores が空でないことを確認してから max() を呼び出す
         if not scores:
             print("❌ ルートカテゴリの推論結果が空でした。パス探索できません。")
             return []

         selected_root_name = max(scores, key=scores.get)
         selected_root_score = scores.get(selected_root_name, 0.0)

         root_selection_threshold = 0.3 # ルート選択の閾値
         if selected_root_score < root_selection_threshold:
              print(f"⚠️ ルートカテゴリの最高スコア ({selected_root_score:.2f}) が低いため、適切なルートが見つかりませんでした。")
              return []

         start_node = next((cat for cat in root_categories if cat.name == selected_root_name), None)
         if not start_node or start_node.id is None:
              print(f"❌ 選ばれたルートカテゴリ '{selected_root_name}' がCategoryオブジェクトリストに見つからないかIDがありません。パス探索できません。")
              return []

         print(f"➡️ 選ばれたルートカテゴリ: '{start_node.name}' (スコア: {selected_root_score:.2f})")
         # 選ばれたルートから再帰的に探索開始
         path_objects = find_path_recursive(start_node.id, [start_node])
         
    elif len(root_categories) == 1:
         # ルートカテゴリが1つの場合はそれから開始
         start_node = root_categories[0]
         if start_node.id is None:
              print("❌ Единственная корневая категория не имеет ID. Путь не может быть построен.") # 日本語に修正
              print("❌ Един一のルートカテゴリにIDがありません。パス探索できません。")
              return []
         # print(f"➡️ Един一のルートカテゴリ: '{start_node.name}' からパス探索開始。") # デバッグ用
         path_objects = find_path_recursive(start_node.id, [start_node])

    else:
         # ルートカテゴリがない場合
         print("❌ ルートカテゴリが見つかりません。パス探索できません。")
         return []

    # 結果を sample_01_rag_construction.py が期待する Dict のリスト形式に変換
    # sample_01_rag_construction.py では {'id': ..., 'name': ..., 'description': ..., 'parent_id': ...} の形式を期待している
    # ここでは、パス上の各カテゴリに対して、その直前のカテゴリのIDを parent_id として設定する形式で Dict を作成する
    # ただし、これは Category オブジェクト自体の parent_ids とは異なる情報になることに注意
    path_dicts = []
    for i, cat_obj in enumerate(path_objects):
         parent_id_in_path = path_objects[i-1].id if i > 0 else None # パス上の直前のノードのID
         path_dicts.append({
             "id": cat_obj.id,
             "name": cat_obj.name,
             "description": cat_obj.description,
             "parent_id": parent_id_in_path # パス上の親のIDを設定 (ツリー描画用)
         })

    return path_dicts


# self-import を削除
# from switch_rag_objects import classify_question_by_llm # 削除済み
# from database import Category # 不要、既に上でインポート済み

def get_probability_tree(
    question: str,
    conn: sqlite3.Connection,
    llm_obj: Any, # LangChain LLM オブジェクトなどを想定
    language: str,
    parent_id: Optional[int] = None, # ルートは None に変更
    threshold: float = 0.00 # 分岐を記録する確率の最小値
) -> List[Dict[str, Any]]: # 戻り値の型ヒントを Dict のリストに修正
    """
    質問に対するカテゴリ確率ツリーを構築する。

    Args:
        question: 質問文
        conn: SQLite 接続
        llm_obj: LLM 推論関数 (LangChain LLM オブジェクトなどを想定)
        language: 質問言語 (ja/en)
        parent_id: 現在の階層の親カテゴリのDB ID。Noneの場合はルートカテゴリ。
        threshold: 分岐を記録する確率の最小値（ノイズ削減のため）

    Returns:
        各カテゴリノードを含むリスト。各ノードは `children` を再帰的に持つ。
        ノードは {'id': ..., 'name': ..., 'description': ..., 'parent_id': ..., 'score': ..., 'children': [...]} の Dict 形式。
    """
    # database モジュールは既に . import database as db とインポートされていることを前提とする

    all_categories = database.Category.get_all_categories(conn)
    if not all_categories:
        # print("⚠️ DBにカテゴリがありません。確率ツリーを構築できません。") # デバッグ用
        return []

    # 親IDに基づいて子カテゴリを見つけるヘルパー関数
    def find_children(parent_cat_id: Optional[int]) -> List[database.Category]:
        """指定された親IDを持つ直接の子カテゴリのリストを返す"""
        children = []
        for cat in all_categories:
             # cat.parent_ids はリスト [int, ...]
             # parent_cat_id が None の場合はルートカテゴリ（parent_ids が空リスト）を探す
             if parent_cat_id is None:
                 if not cat.parent_ids and cat.id is not None: # 親IDリストが空で、IDがNoneでないカテゴリがルート
                      children.append(cat)
             elif cat.id is not None and parent_cat_id is not None and parent_cat_id in cat.parent_ids: # 親IDリストに指定IDが含まれているかチェック
                 children.append(cat)

        # sort_order でソート (必要であれば)
        children.sort(key=lambda c: c.sort_order)
        return children


    # 現在の階層の子カテゴリを取得
    current_level_cats = find_children(parent_id)

    if not current_level_cats:
        # 子カテゴリがいない（ツリーの末端に到達）
        # print(f"⚠️ parent_id={parent_id} の子カテゴリが見つかりませんでした。ツリー構築終了。") # デバッグ用
        return []

    # LLM分類のためにセレクター辞書を構築
    # {カテゴリ名: 説明文} の形式
    selector = {cat.name: cat.description or cat.name for cat in current_level_cats}

    if not selector:
         # 子カテゴリはいるが有効なセレクター情報がない場合 (例: 全て description が None/空)
         # print(f"⚠️ parent_id={parent_id} の子カテゴリに有効なセレクターがありません。ツリー構築終了。") # デバッグ用
         return []

    # LLM に質問文を分類させて、各カテゴリのスコアを取得
    scores = classify_question_by_llm(question, selector, llm_obj, language)

    if not scores:
         # LLM推論が失敗した場合や、classify_question_by_llmが空辞書を返した場合など
         print(f"⚠️ parent_id={parent_id} の階層でカテゴリ推論スコアが取得できませんでした。ツリー構築終了。")
         return []

    # 結果を格納するリスト
    result_tree: List[Dict[str, Any]] = []

    # 現在の階層の各カテゴリについて処理
    for cat in current_level_cats:
        # LLMから得られたスコアを取得 (見つからなければ 0.0)
        score = scores.get(cat.name, 0.0)

        # スコアが閾値以上の場合のみ、ツリーに含め、再帰的に子を探索
        if score >= threshold:
            # 再帰呼び出しでこのカテゴリの子ツリーを構築
            subtree = get_probability_tree(
                question=question,
                conn=conn,
                llm_obj=llm_obj,
                language=language,
                parent_id=cat.id, # 再帰呼び出しでは現在のカテゴリのIDを親IDとして渡す
                threshold=threshold # 同じ閾値を引き継ぐ
            )
            
            # 現在のカテゴリノードの Dict を作成
            # parent_id は、このノードが持つ実際の親IDリストの中から、現在のツリー構築の親として使われたID (parent_id 引数) を設定する
            # もしくは、単に Category オブジェクトが持つ parent_ids リストをそのまま持つ形式でも良い（Dict形式の構造による）
            # sample_01_rag_construction.py の tree_to_graphviz_dot は parent_id (単一整数) を期待しているため、それに合わせる
            # ただし、これは Category オブジェクトの parent_ids とは異なる意味合いになる
            # ここではツリー構造上の親を示す parent_id (引数として渡されたもの) を設定する
            # ルートノードの場合は parent_id は None
            node_dict = {
                "id": cat.id, # CategoryオブジェクトのDB ID
                "name": cat.name,
                "description": cat.description,
                # 注意: ここで設定する parent_id は、Categoryオブジェクト自体の parent_ids とは異なる。
                # これは、このツリー構造における直接の親ノードのIDを示す。
                "parent_id": parent_id, # 再帰呼び出しで渡された親ID
                "score": score,
                "children": subtree
            }
            result_tree.append(node_dict)

    # 結果リストをスコアの高い順にソート (必要であれば)
    # result_tree.sort(key=lambda node: node["score"], reverse=True)

    return result_tree

# --- ツリー構造解析ヘルパー関数 (get_probability_tree の結果を処理するため) ---

def flatten_tree(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    確率ツリー構造をフラットなノードのリストに変換する。
    """
    flat_list = []
    def _recurse(nodes):
        for node in nodes:
            # children キーを除いてコピーするか、そのまま追加
            node_copy = node.copy() # children キーを含むDictをコピー
            flat_list.append(node_copy) # childrenキーも含まれる
            
            # 子がいれば再帰
            if node.get("children"):
                _recurse(node["children"])
    
    _recurse(tree)
    return flat_list

def find_best_leaf_node(flat_nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    フラットなノードリストから、最もスコアの高い「葉ノード」を見つける。
    葉ノードとは、children が空リスト [] または存在しないノード。
    """
    best_node = None
    highest_score = -1.0 # スコアは0以上なので初期値-1.0でOK

    for node in flat_nodes:
        # children が空リストまたはキー自体が存在しないか確認
        is_leaf = not node.get("children") # children が None, [], False などなら True になる
        
        if is_leaf:
            score = node.get("score", 0.0)
            # スコアが現在の最高スコアより高い場合、更新
            if score > highest_score:
                highest_score = score
                best_node = node

    # スコアが0以下の場合は有効な葉ノードとみなさないなどの閾値設定も考慮可能
    # 例: if best_node and best_node.get("score", 0.0) > 0: return best_node else return None
    # ここではシンプルに最高スコアノードを返す
    # sample_01_rag_construction.py 側でスコア閾値チェックを行っているため、ここではそのまま返す
    return best_node


# --- テスト用関数 ---
# if __name__ == "__main__": の中に移動して、database.py のテストと統合するのが良い

# def test_get_category_path():
#     print("Testing get_category_path...")
#     # DB接続とダミーデータの準備が必要
#     # llm_obj のダミー実装も必要
#     pass

# def test_get_probability_tree():
#     print("Testing get_probability_tree...")
#     # DB接続とダミーデータの準備が必要
#     # llm_obj のダミー実装も必要
#     pass

# if __name__ == "__main__":
#     # ダミーのDB接続とLLMオブジェクトを作成
#     class DummyConnection:
#          # ... (database.py のテストコードにある DummyConnection と DummyCursor をコピー)
#          def cursor(self):
#               return DummyCursor()
#          def commit(self):
#               pass
#          def rollback(self):
#               pass
#          def close(self):
#               pass

#     class DummyCursor:
#          # ... (database.py のテストコードにある DummyCursor をコピー)
#          def execute(self, sql, params=None):
#               print(f"DEBUG DB: Executing {sql} with {params}")
#               # ダミーのカテゴリデータを返すロジック
#               # Category.get_all_categories は Category オブジェクトのリストを返す想定
#               # ダミーカーソルは row (タプル) を返すが、Category.from_row でオブジェクト化される想定でデータを用意
#               if "SELECT id, name, description, type_code, sort_order, created_at, updated_at FROM categories" in sql:
#                    # id, name, description, type_code, sort_order, created_at, updated_at (7カラム)
#                    self._rows = [
#                         (1, "情報", "コンピューター関連", "hier", 10, "now", "now"),
#                         (2, "土木", "建設関連", "hier", 20, "now", "now"),
#                         (3, "医療", "医療関連", "hier", 30, "now", "now"),
#                         (4, "大学", "大学関連", "hier", 40, "now", "now"),
#                         (5, "学生", "学生生活関連", "hier", 50, "now", "now"), # 大学の子としてテスト
#                         (6, "重要", "重要な文書", "flat", 100, "now", "now"),
#                         (999, "その他", "上記のどれにも明確に分類されない場合", "flat", 999, "now", "now"), # その他カテゴリもDBに存在すると仮定
#                    ]
#               elif "SELECT parent_category_id FROM category_parents WHERE child_category_id = ?" in sql:
#                    # ダミーの親子関係データ
#                    child_id = params[0]
#                    if child_id == 5: # 学生カテゴリ(ID=5)の子
#                         self._rows = [(4,)] # 親は大学(ID=4)
#                         # 複数親のテスト: [(4,), (1,)] # 親は大学(ID=4)と情報(ID=1)
#                    elif child_id in (1, 2, 3, 4, 6, 999): # ルートカテゴリには親がいない
#                         self._rows = []
#                    else:
#                         self._rows = [] # 未知のカテゴリの子
#               # Category.get_by_id で親カテゴリの名前を取得するために SELECT name FROM categories WHERE id = ? が呼ばれる可能性
#               elif sql.strip().startswith("SELECT name FROM categories WHERE id = ?"):
#                    parent_id_to_find_name = params[0]
#                    dummy_cat_names = {1: "情報", 2: "土木", 3: "医療", 4: "大学", 5: "学生", 6: "重要", 999: "その他"}
#                    name_found = dummy_cat_names.get(parent_id_to_find_name)
#                    self._rows = [(name_found,)] if name_found else []
#               elif "SELECT type_code, type_name FROM category_types" in sql: # init_tablesで使うかも
#                    self._rows = [("hier", "階層型"), ("flat", "フラット型"), ("array", "配列型")]
#               elif sql.strip().startswith("SELECT id, name, description") and "WHERE id=?" in sql: # Category.get_by_idで使う
#                    cat_id = params[0]
#                    # ダミーカテゴリデータからIDで検索して1件返す
#                    dummy_cats_data = [
#                         (1, "情報", "コンピューター関連", "hier", 10, "now", "now"),
#                         (2, "土木", "建設関連", "hier", 20, "now", "now"),
#                         (3, "医療", "医療関連", "hier", 30, "now", "now"),
#                         (4, "大学", "大学関連", "hier", 40, "now", "now"),
#                         (5, "学生", "学生生活関連", "hier", 50, "now", "now"),
#                         (6, "重要", "重要な文書", "flat", 100, "now", "now"),
#                         (999, "その他", "上記のどれにも明確に分類されない場合", "flat", 999, "now", "now"),
#                    ]
#                    found_row = next((row for row in dummy_cats_data if row[0] == cat_id), None)
#                    self._rows = [found_row] if found_row else []
#               else:
#                    print(f"DEBUG DB: Unhandled query: {sql}")
#                    self._rows = [] # その他のクエリは空結果
#               self._index = 0

#          def fetchone(self):
#               if self._index < len(self._rows):
#                    row = self._rows[self._index]
#                    self._index += 1
#                    return row
#               return None

#          def fetchall(self):
#               rows = self._rows[self._index:]
#               self._index = len(self._rows)
#               return rows

#          def lastrowid(self): # insertテスト用
#               return 1 # ダミー

#          def executemany(self, sql, params_list): # executemanyテスト用
#               print(f"DEBUG DB: Executing executemany {sql} with {len(params_list)} sets of params")
#               pass # 何もしない


#     # Dummy LLM Object (invokeメソッドを持つ想定)
#     # 質問とセレクターを受け取り、{匿名ラベル: 確率} の辞書を返す（JSON文字列形式で）
#     class DummyLLM:
#          def invoke(self, prompt):
#               print(f"\n--- DEBUG LLM Prompt ---")
#               print(prompt[:1000] + "...") # 長すぎるので一部表示
#               print("--- End Prompt ---")

#               # プロンプトから匿名ラベルとカテゴリ名を抽出して適当な確率を生成する
#               # プロンプト形式に依存する。`{f"C{i+1}": (name, desc)}` の形式で label_map を内部で再構築する必要がある。
#               # これは classify_question_by_llm の内部ロジックをある程度シミュレーションすることになる。
#               # より簡単な方法は、LLMに渡されるセレクター({カテゴリ名: 説明})を引数として受け取るように DummyLLM を設計すること。
#               # classify_question_by_llm から DummyLLM.invoke() が呼ばれる際、引数はprompt文字列のみ。
#               # なので、prompt 文字列からセレクター情報をパースする必要がある。これは難しい。
#               # DummyLLM は classify_question_by_llm の外部ではなく、内部で使用される classify_question_by_llm_internal として設計し、
#               # LLM呼び出し部分だけを置き換える方がテストしやすいかもしれない。
#               # ここでは classify_question_by_llm のプロンプト構造を前提としてパースを試みる。

#               label_map_in_prompt = {}
#               # 例: "- C1: コンピューター関連" の形式をパース
#               selector_lines = re.findall(r'- (C\d+): (.*)', prompt)
#               for label, desc in selector_lines:
#                    # ここでは名前がわからないので、匿名ラベルと説明だけ
#                    label_map_in_prompt[label] = (label, desc) # 仮の名前としてラベルを使うか、ダミー名

#               # その他カテゴリの匿名ラベルと説明を探す
#               other_match = re.search(r'- (C\d+): 上記のどれにも明確に分類されない場合', prompt)
#               if other_match:
#                    other_label = other_match.group(1)
#                    label_map_in_prompt[other_label] = ("その他", other_match.group(2)) # カテゴリ名を「その他」とする


#               question_match = re.search(r'質問文：\s*([\s\S]*?)(?:\n\n|\Z)', prompt)
#               question_part = question_match.group(1).strip() if question_match else ""

#               scores = {}
#               # 簡単なキーワードマッチでスコアを割り当てる
#               if "大学" in question_part:
#                    for label, (name, desc) in label_map_in_prompt.items():
#                         if "大学" in desc or "大学" in name:
#                              scores[label] = scores.get(label, 0) + 0.7 # descまたは名前に大学があれば加点
#                         if "学生" in desc or "学生" in name:
#                              scores[label] = scores.get(label, 0) + 0.2 # 学生も関連
#                         if "制度" in question_part and ("制度" in desc or "制度" in name):
#                              scores[label] = scores.get(label, 0) + 0.1 # 制度があればさらに加点

#               elif "学生" in question_part:
#                    for label, (name, desc) in label_map_in_prompt.items():
#                         if "学生" in desc or "学生" in name:
#                              scores[label] = scores.get(label, 0) + 0.8
#                         if "大学" in desc or "大学" in name:
#                              scores[label] = scores.get(label, 0) + 0.15

#               elif "医療" in question_part:
#                    for label, (name, desc) in label_map_in_prompt.items():
#                         if "医療" in desc or "医療" in name:
#                              scores[label] = scores.get(label, 0) + 0.95
#               elif "土木" in question_part:
#                     for label, (name, desc) in label_map_in_prompt.items():
#                         if "建設" in desc or "土木" in desc or "土木" in name:
#                              scores[label] = scores.get(label, 0) + 0.95
#               else:
#                    # その他に加点
#                    for label, (name, desc) in label_map_in_prompt.items():
#                         if name == "その他":
#                              scores[label] = scores.get(label, 0) + 0.7
#                         else:
#                              scores[label] = scores.get(label, 0) + 0.05 # その他以外に少し配点

#               # 全てのラベルが結果に含まれるように（LLMの出力形式に合わせる）
#               final_scores = {label: scores.get(label, 0.0) for label in label_map_in_prompt.keys()}


#               # 結果をJSON形式文字列に変換 (LLMが出力するであろう形式をシミュレーション)
#               json_output = json.dumps(final_scores, indent=2, ensure_ascii=False)
#               # 応答形式を simulate (```json ... ``` コードブロックなし)
#               response_text = json_output 
#               # response_text = f"以下が分類結果です。\n```json\n{json_output}\n```\n" # コードブロックありの例

#               print(f"\n--- DEBUG LLM Response ---")
#               print(response_text)
#               print("--- End Response ---")
#               # invoke は Message オブジェクトを返す想定
#               class DummyMessage:
#                    def __init__(self, content): self.content = content
#               return DummyMessage(response_text)


#     dummy_conn = DummyConnection()
#     dummy_llm = DummyLLM()

#     print("\n--- Testing get_probability_tree ---")
#     # question_for_tree = "大学の制度に関する質問"
#     question_for_tree = "大学生のアルバイトについて知りたい" # 学生カテゴリに誘導される質問
#     tree_result = get_probability_tree(question_for_tree, dummy_conn, dummy_llm, language="ja", threshold=0.0) # thresholdを低くしてツリー全体を見る
#     print(f"\n--- Probability Tree Result for '{question_for_tree}' ---")
#     import yaml # YAML形式で出力すると見やすい
#     print(yaml.dump(tree_result, allow_unicode=True, default_flow_style=False, indent=2))

#     # flatten_tree と find_best_leaf_node のテスト
#     print("\n--- Testing flatten_tree and find_best_leaf_node ---")
#     flat_nodes = flatten_tree(tree_result)
#     print(f"Flattened nodes count: {len(flat_nodes)}")
#     # for node in flat_nodes:
#     #      print(node) # デバッグ用

#     best_leaf = find_best_leaf_node(flat_nodes)
#     print(f"Best leaf node: {best_leaf}")
#     # 期待される結果例: {'id': 5, 'name': '学生', ..., 'score': 0.8, 'children': []} (dummy_llm_fn のロジックによる)


#     print("\n--- Testing get_category_path ---")
#     question_for_path = "大学生のバイトについて知りたい" # get_probability_tree と同じ質問でテスト
#     path_result = get_category_path(question_for_path, dummy_conn, dummy_llm, language="ja")
#     print(f"\n--- Category Path Result for '{question_for_path}' ---")
#     print(path_result)
#     # 期待される結果例 (greedy なので scores 次第だが):
#     # ルートで大学(ID=4)が選ばれ、その子で学生(ID=5)が選ばれれば
#     # [{'id': 4, 'name': '大学', 'description': '...', 'parent_id': None}, {'id': 5, 'name': '学生', 'description': '...', 'parent_id': 4}]


#     print("\n--- Testing classify_question_by_llm ---")
#     dummy_selector = {
#          "カテゴリA": "Aに関する説明",
#          "カテゴリB": "Bに関する説明",
#          "その他": "その他の場合"
#     }
#     question_for_classify = "これはカテゴリAについてです。"
#     classification_result = classify_question_by_llm(question_for_classify, dummy_selector, dummy_llm, language="ja")
#     print(f"\n--- Classify Question Result for '{question_for_classify}' ---")
#     print(classification_result)
#     # 期待される結果例: {'カテゴリA': 0.9, 'カテゴリB': 0.05, 'その他': 0.05} (dummy_llm_fn のロジックによる)