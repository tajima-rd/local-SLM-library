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
    
def vectorization(in_file, md_path, vect_path, category, overwrite=False, embedding_name="nomic-embed-text:latest"):
    # 形式を確認して処理します
    print("※ granite-embedding:278m を選ぶと処理が落ちます…")
    # embedding_name = select_embedding_model()
    embedding = EmbeddingWrapper( # type: ignore
        name=embedding_name,
        model=load_embedding_model(name=embedding_name)
    )

    # 入力ファイルとドキュメント形式の確認
    doc_path = Path(in_file)
    doc_format = rag.get_document_format(doc_path)

    if not doc_format:
        print(f"サポートされていないドキュメント形式: {doc_path.suffix}")
        exit()

    # Markdown形式ならコピー、それ以外は変換
    if doc_format == InputFormat.MD:
        shutil.copy2(doc_path, md_path)
    elif not os.path.exists(md_path):
        rag.convert_document_to_markdown(doc_path, md_path)

    # ベクトルストア構築
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




