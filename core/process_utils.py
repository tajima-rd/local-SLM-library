from pathlib import Path
from .document_utils import convert_document_to_markdown
from .ingestion import save_markdown_to_vectorstore
from .retriever_utils import RetrieverCategory

def process_and_vectorize_file(
    in_file: Path,
    markdown_dir: Path,
    vectorstore_dir: Path,
    category: RetrieverCategory,
    embedding_name: str = "nomic-embed-text:latest",
    overwrite: bool = True
):
    """
    単一ファイルを変換・ベクトルストア化する処理を統合。
    """
    # Markdown変換
    if in_file.suffix.lower() == ".md":
        md_path = in_file
    else:
        try:
            md_path = convert_document_to_markdown(
                in_file, markdown_dir / in_file.with_suffix(".md").name
            )
        except ValueError as e:
            print(f"⚠️ スキップ（変換失敗）: {e}")
            return None

    if not md_path or not md_path.exists():
        print(f"⚠️ スキップ（無効なファイル）: {md_path}")
        return None

    vect_path = vectorstore_dir / f"{md_path.stem}.faiss"
    print(f"📦 ベクトルストアを構築します: {md_path.name}")

    return save_markdown_to_vectorstore(
        md_path=md_path,
        vect_path=vect_path,
        category=category,
        overwrite=overwrite,
        embedding_name=embedding_name
    )
