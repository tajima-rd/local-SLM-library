# 必要なパッケージをインストールします
import os, shutil
# from modules.core import rag
import rag
from pathlib import Path 
from docling.datamodel.base_models import InputFormat  # type: ignore
from langchain_ollama import OllamaEmbeddings # type: ignore
import requests # type: ignore
from dataclasses import dataclass

@dataclass
class EmbeddingWrapper:
    name: str  # モデル名（"nomic-embed-text:latest" など）
    model: OllamaEmbeddings  # LangChainの埋め込みモデルオブジェクト


def get_embedding_models(url="http://localhost:11434/api/tags"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
        
        embedding_models = [
            model["name"] for model in models
            if "embed" in model["name"].lower() or "embedding" in model["name"].lower()
        ]

        if not embedding_models:
            print("埋め込みモデルが見つかりませんでした。")
            return []

        print("利用可能な埋め込みモデル一覧：")
        for name in embedding_models:
            print(f" - {name}")

        return embedding_models

    except requests.exceptions.ConnectionError:
        print("Ollama サーバに接続できません。`ollama serve` を起動していますか？")
        return []
    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")
        return []


def select_embedding_model():
    models = get_embedding_models()
    if not models:
        print("モデルが見つからなかったため、デフォルトを使用します。")
        return "nomic-embed-text:latest"

    print("\n 使用する埋め込みモデルを番号で選択してください：")
    for i, name in enumerate(models):
        print(f" [{i}] {name}")

    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("無効な番号です。再入力してください。")
        except ValueError:
            print(" 数字を入力してください。")

def load_embedding_model(name = "nomic-embed-text:latest"):
    return OllamaEmbeddings(model=name)

def construct(dir_markdown, dir_vectorhouse, file_list, overwrite=False, embedding_name="nomic-embed-text:latest"):
    print("この関数のサポートを辞めました。")
    
def vectorization(md_path, vect_path, category, overwrite=False, embedding_name="nomic-embed-text:latest"):
    """
    Markdownファイルをもとにベクトルストアを生成する関数。
    入力が .md 形式以外の場合は処理を中断します。
    """
    md_path = Path(md_path)
    
    # .md ファイルであることを確認
    if md_path.suffix.lower() != ".md":
        raise ValueError(f"入力ファイルは Markdown (.md) 形式である必要があります: {md_path}")

    print("※ granite-embedding:278m を選ぶと処理が落ちます…")

    embedding = EmbeddingWrapper(  # type: ignore
        name=embedding_name,
        model=load_embedding_model(name=embedding_name)
    )

    if overwrite or not os.path.exists(vect_path):
        vectorstore = rag.save_chain(
            md_path,
            vect_path,
            embedding,
            category,
            loader_type="markdown",
            text_splitter_type="recursive"
        )
        return vectorstore
    else:
        print(f"既存のベクトルストアが存在します: {vect_path}（上書きしません）")
        return None