import csv
import sys
from pathlib import Path
import inspect

from core import document_utils as du
from core import database # type: ignore

class Stage:
    """
    膵がんの各ステージに関する情報を格納するクラス。

    Attributes:
        stage (str): ステージ名 (例: "Stage 0", "Stage IA" など)。
        examination_result (str): そのステージにおける検査結果に関する詳細な説明。
        subjective_symptoms (str): そのステージにおける自覚症状に関する詳細な説明。
        lifestyle (str): そのステージにおける生活習慣との関連に関する詳細な説明。
        medical_history (str): そのステージにおける既往歴との関連に関する詳細な説明。
        family_history (str): そのステージにおける家族歴との関連に関する詳細な説明。
    """

    def __init__(self, stage: str, examination_result: str, subjective_symptoms: str, lifestyle: str, medical_history: str, family_history: str):
        """
        Stage クラスの新しいインスタンスを初期化します。

        Args:
            stage (str): ステージ名。
            examination_result (str): 検査結果の説明。
            subjective_symptoms (str): 自覚症状の説明。
            lifestyle (str): 生活習慣との関連の説明。
            medical_history (str): 既往歴との関連の説明。
            family_history (str): 家族歴との関連の説明。
        """
        self.stage = stage
        self.examination_result = examination_result
        self.subjective_symptoms = subjective_symptoms
        self.lifestyle = lifestyle
        self.medical_history = medical_history
        self.family_history = family_history

    def __str__(self):
        """
        Stage オブジェクトを人間が読める文字列として表現します。
        """
        # 各属性のテキストが長い場合があるので、適宜改行などを入れると見やすいかもしれません
        return (
            f"--- ステージ: {self.stage} ---\n"
            f"検査結果: {self.examination_result}\n"
            f"自覚症状: {self.subjective_symptoms}\n"
            f"生活習慣: {self.lifestyle}\n"
            f"既往歴: {self.medical_history}\n"
            f"家族歴: {self.family_history}\n"
            "--------------------------"
        )

    def __repr__(self):
        """
        Stage オブジェクトを開発者向けの文字列として表現します。
        （テキストが長いので、一部を省略して表示します）
        """
        return (
            f"Stage(stage='{self.stage}',\n"
            f"      examination_result='{self.examination_result[:100]}...', \n"
            f"      subjective_symptoms='{self.subjective_symptoms[:100]}...', \n"
            f"      lifestyle='{self.lifestyle[:100]}...', \n"
            f"      medical_history='{self.medical_history[:100]}...', \n"
            f"      family_history='{self.family_history[:100]}...')\n"
        )

def get_csv_data(csv_file_path):
    """
    指定されたCSVファイルからデータを読み込み、Stageクラスのオブジェクトのリストとして返します。

    CSVファイルは以下の構造を想定しています。
    - 1行目がヘッダーで、最初のカラムは「特徴の軸」、2列目以降がステージ名。
    - 2行目以降がデータ行で、最初のカラムは「特徴の軸」、2列目以降が各ステージの説明テキスト。
    - 行の順序は不定でも処理可能です（1列目の「特徴の軸」をキーとします）。

    Args:
        csv_file_path (str): 読み込むCSVファイルへのパス。

    Returns:
        list[Stage]: Stageクラスのオブジェクトのリスト。
                     ファイルの読み込みや処理に失敗した場合、
                     またはCSVの構造や特徴の軸が期待通りでない場合は空のリストを返します。
    """
    raw_data = []
    try:
        # ファイルを読み込みモードで開く (エンコーディングをutf-8と仮定)
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                 # 行の末尾の空要素を削除（CSVによっては不要なカンマで空要素ができる場合があるため）
                while row and row[-1] == '':
                    row.pop()
                raw_data.append(row)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {csv_file_path}")
        return []
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return []

    # 基本的な構造チェック
    if not raw_data or len(raw_data) < 1 or not raw_data[0] or raw_data[0][0] != '特徴の軸':
        print("エラー: CSVデータが不正な形式です（ヘッダーまたは最初のカラムが無効）。")
        return []

    header_row = raw_data[0]
    stage_names = header_row[1:] # Stage 0, Stage IA, ... のリスト

    if not stage_names:
        print("エラー: CSVヘッダー行にステージ名が見つかりません。")
        return []

    # Stageクラスの属性名に対応させるためのマッピング
    # CSVファイルの特徴の軸名とStageクラスの属性名が一致しない場合、ここでマッピングを定義
    # ここに定義されていない特徴の軸は無視されます。
    feature_name_to_attribute = {
        '検査結果': 'examination_result',
        '自覚症状': 'subjective_symptoms',
        '生活習慣': 'lifestyle',
        '既往歴': 'medical_history',
        '家族歴': 'family_history',
        # 必要に応じて他の特徴を追加
    }

    # Stageクラスのコンストラクタが必要とする属性名を取得 (self を除く)
    # Stageクラスの定義がこの関数の前に必要です。
    try:
        stage_init_params = list(inspect.signature(Stage.__init__).parameters.keys())
        stage_init_params.remove('self')
    except NameError:
        print("エラー: Stageクラスが定義されていません。")
        return []
    except Exception as e:
        print(f"Stageクラスのコンストラクタ引数の取得中にエラーが発生しました: {e}")
        return []

    # Stageクラスの__init__が受け取る属性（'stage'を含む）全てが、
    # マッピング辞書の値として定義されているか、または 'stage' 自体であるかを確認
    mapped_attributes = set(feature_name_to_attribute.values())
    missing_required_mappings = [
        param for param in stage_init_params
        if param != 'stage' and param not in mapped_attributes
    ]
    if missing_required_mappings:
        print(f"エラー: Stageクラスのコンストラクタに必要な以下の属性に対応するCSV特徴軸名がマッピング辞書に定義されていません: {missing_required_mappings}")
        print(f"  マッピング辞書 '{feature_name_to_attribute}' を確認してください。")
        return [] # 必須属性のマッピングがない場合は処理を中断


    # --- ユーザーが求めた「1列目をキーとする辞書」に関連する処理の開始 ---

    # データ行を処理し、「特徴の軸」をキーとする辞書に格納
    # feature_data_by_name は、行の順序に関係なく特徴名でデータを参照できるようにするためのものです。
    # 例: {'検査結果': ['Stage 0 検査結果テキスト', 'Stage IA 検査結果テキスト', ...], ...}
    feature_data_by_name = {}
    for row_index, row in enumerate(raw_data[1:]): # ヘッダー行を除いてループ (row_index は raw_data[1:] に対するインデックス)
        # 空行や、最初のカラムが空の行はスキップ
        if not row or not row[0]:
            continue

        feature_csv_name = row[0] # CSVの1列目の特徴名
        feature_values = row[1:]  # その行の各ステージに対応する値のリスト

        # feature_name_to_attribute に存在しない特徴名は無視
        if feature_csv_name not in feature_name_to_attribute:
             # print(f"警告: 未知の特徴の軸 '{feature_csv_name}' (CSV行 {row_index+2}) はマッピング辞書に定義されていません。スキップします。")
            continue

        # ステージの数と行のデータ数が一致するか確認 (重要)
        if len(feature_values) != len(stage_names):
            print(f"警告: 特徴 '{feature_csv_name}' (CSV行 {row_index+2}) のデータ数がヘッダーのステージ数と一致しません ({len(feature_values)} vs {len(stage_names)})。この行のデータをスキップします。")
            continue

        # マッピングに存在する特徴名と、その行の全ステージのデータを辞書に格納
        feature_data_by_name[feature_csv_name] = feature_values

    # feature_data_by_name はこれで完成。行の順序は関係なくなった。

    # feature_data_by_name 辞書を使って、ステージごとのデータ（Stageオブジェクト作成用の形式）を組み立てる
    # 例: {'Stage 0': {'stage': 'Stage 0', 'examination_result': '...', 'subjective_symptoms': '...', ...}, ...}
    stage_data_dict = {}
    for i, stage_name in enumerate(stage_names): # ステージ名の順番にループ
        # Stageクラスの__init__に渡す引数を格納する辞書を作成
        # Stageクラスの__init__が要求する全ての引数名をキーとして初期化
        stage_data_for_init = {'stage': stage_name} # 'stage' は別途追加

        is_stage_data_complete = True # このステージのデータがStageオブジェクト作成に十分か

        # Stageクラスのコンストラクタ引数（'self'を除く）をループ
        for param_name in stage_init_params:
            if param_name == 'stage':
                # 'stage' は既に stage_data_for_init に追加済み
                continue

            # このStage属性(param_name)に対応するCSVの特徴軸名を見つける（マッピング辞書を逆引き）
            feature_csv_name = None
            for csv_name, attr_name in feature_name_to_attribute.items():
                if attr_name == param_name:
                    feature_csv_name = csv_name
                    break # 見つかったらループを抜ける

            # feature_csv_name は必須マッピングチェックで存在が確認済みのはずだが、念のため
            if feature_csv_name is None:
                 # ここに来ることは通常ない想定（必須マッピングチェックで弾かれるため）
                 print(f"内部エラー: Stageクラスの属性 '{param_name}' に対応するCSV特徴軸名がマッピング辞書に見つかりません。")
                 is_stage_data_complete = False
                 stage_data_for_init[param_name] = ""
                 continue


            # feature_data_by_name 辞書から、この特徴軸に対応する行データ（全ステージ分）を取得
            if feature_csv_name in feature_data_by_name:
                feature_values = feature_data_by_name[feature_csv_name]
                # 現在のステージ(インデックスi)に対応するテキストを取得
                if i < len(feature_values):
                    stage_data_for_init[param_name] = feature_values[i]
                else:
                    # インデックス範囲外（これは通常、行データ数がステージ数と一致しない場合の警告で捕捉されるはず）
                    print(f"警告: 特徴 '{feature_csv_name}' のデータがステージ '{stage_name}' ({i}番目) に対して不足しています。空文字列を割り当てます。")
                    stage_data_for_init[param_name] = ""
                    is_stage_data_complete = False # データ不足なので不完全とマーク
            else:
                 # feature_data_by_name にこの特徴軸自体が存在しない
                 # （マッピングにはあるが、対応するCSVデータ行がスキップされた、またはCSVに存在しなかったケースなど）
                 print(f"警告: Stageクラスの属性 '{param_name}' に対応するCSV特徴軸 '{feature_csv_name}' のデータがCSVから読み込めませんでした（ステージ '{stage_name}'）。")
                 is_stage_data_complete = False
                 stage_data_for_init[param_name] = "" # データなしとして空文字列をセット


        # このステージのデータが Stage オブジェクト作成に十分か最終確認
        # is_stage_data_complete が False の場合は、必須属性のデータが不足している
        if not is_stage_data_complete:
             # 不足している具体的な属性名は、stage_data_for_init に空文字列がセットされたかどうかで判断可能
             missing_attrs = [attr for attr, value in stage_data_for_init.items() if value == "" and attr != 'stage']
             if missing_attrs:
                 print(f"警告: ステージ '{stage_name}' のデータが不完全です。Stageオブジェクト作成に必要な属性が不足しています: {missing_attrs} (Stageクラスのコンストラクタ引数名)")
                 # このステージのオブジェクトは作成しない
                 continue # このステージの処理をスキップ


        # Stageオブジェクトを作成
        try:
            # stage_data_for_init 辞書には、Stageクラスの__init__が受け取る全ての引数名がキーとして含まれている状態になっているはず
            # 辞書を展開してキーワード引数として渡す
            stage_instance = Stage(**stage_data_for_init)
            stage_data_dict[stage_name] = stage_instance # ステージ名をキーとしてStageインスタンスを格納
        except TypeError as e:
            # __init__に不要な引数が渡された、または引数名が一致しないなどのTypeError
            print(f"エラー: ステージ '{stage_name}' のStageオブジェクト作成に失敗しました（TypeError）。Stageクラスのコンストラクタ引数とデータが一致しませんか？: {e}")
            print(f"  データ keys() provided: {stage_data_for_init.keys()}")
            print(f"  Stage __init__ params: {stage_init_params}")
            # エラーメッセージを出力し、このステージはスキップ
            continue
        except Exception as e:
            print(f"エラー: ステージ '{stage_name}' のStageオブジェクト作成中に予期しないエラーが発生しました: {e}")
            # エラーメッセージを出力し、このステージはスキップ
            continue

    # --- ユーザーが求めた「1列目をキーとする辞書」に関連する処理の終了 ---


    # 最終的な Stage オブジェクトのリストは、ステージ名の順序に従って構築
    stage_objects = [stage_data_dict[name] for name in stage_names if name in stage_data_dict]

    return stage_objects


if __name__ == "__main__":
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
    stage_file_path = scenario_dir / "stage.csv" # Adjust path if needed

    print("--- Running Stage Definition Script ---")

    # Check if the sample file exists
    if not stage_file_path.exists():
        print(f"Error: Sample file not found at {stage_file_path}")
        print("Please place a sample .amua file in the ./sample directory or update the file_path.")
        sys.exit(1)

    # When the script is executed directly, call the main function
    stage_list = get_csv_data(stage_file_path)
    for stage in stage_list:
        print(stage)
    
    # Step 0: データベースを構築する
    database.init_db(db_path, overwrite=True)
    conn = database.db_connect(db_path)
    if conn is None:
        print("Error: Could not connect to the database.")
        sys.exit(1)
    print("✅ Database initialized successfully.")

    # Step 1: Project オブジェクトの生成
    new_project = database.Project(
        name="Decision Support for Pancreatic Cancer Staging",
        description="膵がんのステージングに関する情報を提供し、患者の診断と治療計画を支援するためのプロジェクト。",
        author="藤本悠",
        status="active",
        default_model_name="granite3.3:2b",
        default_prompt_name="japanese_concise",
        default_embedding_name="bge-m3",
        notes="実装のテスト",
        dbcon=conn, 
        insert=True
    )

    cat_stage = database.Category(
        name = "ステージング",
        description = "膵がんのステージングに関するカテゴリー",
        type_code = "hier",
        sort_order = 0,
        dbcon=conn, 
        insert=True
    )

    cat_exam = database.Category(
        name = "検査結果",
        description = "膵がんのステージングにおける検査結果に関するカテゴリー",
        parent_ids = [cat_stage.id],
        type_code = "hier",
        sort_order = 0,
        dbcon=conn, 
        insert=True
    )

    cat_subj = database.Category(
        name = "自覚症状",
        description = "膵がんのステージングにおける自覚症状に関するカテゴリー",
        parent_ids = [cat_stage.id],
        type_code = "hier",
        sort_order = 0,
        dbcon=conn, 
        insert=True
    )

    cat_life = database.Category(
        name = "生活習慣",
        description = "膵がんのステージングにおける生活習慣に関するカテゴリー",
        parent_ids = [cat_stage.id],
        type_code = "hier",
        sort_order = 0,
        dbcon=conn, 
        insert=True
    )

    cat_medh = database.Category(
        name = "既往歴",
        description = "膵がんのステージングにおける既往歴に関するカテゴリー",
        parent_ids = [cat_stage.id],
        type_code = "hier",
        sort_order = 0,
        dbcon=conn, 
        insert=True
    )

    cat_famh = database.Category(
        name = "家族歴",
        description = "膵がんのステージングにおける家族歴に関するカテゴリー",
        parent_ids = [cat_stage.id],
        type_code = "hier",
        sort_order = 0,
        dbcon=conn, 
        insert=True
    )

        # Stage 0 のカテゴリを作成
    cat_stage0 = database.Category(
        name = "Stage 0",
        description = "膵がん Stage 0 に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 1, # ステージ順に並べるためのソート順 (親が0なら子は1から始めるなど)
        dbcon=conn, # DB接続オブジェクト
        insert=True # 作成時にDBに挿入
    )

    # Stage IA のカテゴリを作成
    cat_stage1a = database.Category(
        name = "Stage IA",
        description = "膵がん Stage IA に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 2, # ステージ順に並べるためのソート順
        dbcon=conn,
        insert=True
    )

    # Stage IB のカテゴリを作成
    cat_stage1b = database.Category(
        name = "Stage IB",
        description = "膵がん Stage IB に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 3, # ステージ順に並べるためのソート順
        dbcon=conn,
        insert=True
    )

    # Stage IIA のカテゴリを作成
    cat_stage2a = database.Category(
        name = "Stage IIA",
        description = "膵がん Stage IIA に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 4, # ステージ順に並べるためのソート順
        dbcon=conn,
        insert=True
    )

    # Stage IIB のカテゴリを作成
    cat_stage2b = database.Category(
        name = "Stage IIB",
        description = "膵がん Stage IIB に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 5, # ステージ順に並べるためのソート順
        dbcon=conn,
        insert=True
    )

    # Stage III のカテゴリを作成
    cat_stage3 = database.Category(
        name = "Stage III",
        description = "膵がん Stage III に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 6, # ステージ順に並べるためのソート順
        dbcon=conn,
        insert=True
    )
 
    # Stage IV のカテゴリを作成
    cat_stage4 = database.Category(
        name = "Stage IV",
        description = "膵がん Stage IV に関するカテゴリー",
        parent_ids = [cat_stage.id], # 親カテゴリのIDをリストで指定
        type_code = "hier",
        sort_order = 7, # ステージ順に並べるためのソート順
        dbcon=conn,
        insert=True
    )