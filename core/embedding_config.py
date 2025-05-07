# embedding_config.py

import requests  # type: ignore
from langchain_ollama import OllamaEmbeddings  # type: ignore

def get_embedding_models(url: str = "http://localhost:11434/api/tags") -> list[str]:
    """
    Ollama API から使用可能な埋め込みモデル名を取得します。

    Parameters:
    - url: Ollama API の tags エンドポイント

    Returns:
    - モデル名のリスト（"embed" を含むもの）
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])

        return [
            model["name"] for model in models
            if "embed" in model["name"].lower() or "embedding" in model["name"].lower()
        ]

    except requests.exceptions.ConnectionError:
        print("❌ Ollama サーバに接続できません。`ollama serve` が起動しているか確認してください。")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ エラーが発生しました: {e}")
        return []

def select_embedding_model() -> str:
    """
    使用可能な埋め込みモデルを一覧表示し、ユーザーが番号で選択できるようにします。

    Returns:
    - 選択されたモデル名（str）
    """
    models = get_embedding_models()
    if not models:
        print("⚠️ モデルが見つからなかったため、デフォルト 'nomic-embed-text:latest' を使用します。")
        return "nomic-embed-text:latest"

    print("\n使用可能な埋め込みモデル一覧：")
    for i, name in enumerate(models):
        print(f" [{i}] {name}")

    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("無効な番号です。もう一度入力してください。")
        except ValueError:
            print("⚠️ 数字を入力してください。")

def load_embedding_model(name: str = "nomic-embed-text:latest") -> OllamaEmbeddings:
    """
    指定された埋め込みモデル名に基づいて OllamaEmbeddings インスタンスを作成します。

    Parameters:
    - name: モデル名（例: "nomic-embed-text:latest"）

    Returns:
    - OllamaEmbeddings インスタンス
    """
    return OllamaEmbeddings(model=name)
