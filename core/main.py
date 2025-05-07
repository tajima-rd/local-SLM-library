from pathlib import Path

from langchain_ollama import OllamaLLM # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore 

from RAGSession import RAGSession  # type: ignore
from retriever_utils import HierarchicalRetrieverCategory


# --- ディレクトリ設定 ---
# 現在のファイル（スクリプト）のパス
current_path = Path(__file__).resolve()
 
# 'core' ディレクトリを含む親ディレクトリを見つける
core_root = next(p for p in current_path.parents if p.name == "core")

# そこから目的のサブパスを定義
base_dir = core_root / "sample"
markdown_dir = base_dir / "markdown"
vectorstore_dir = base_dir / "vectorstore"

# --- 入力ファイル ---
file_category_pairs = [
    (markdown_dir / "cat_tools.md", HierarchicalRetrieverCategory(tagname="情報", level=1)),
    (markdown_dir / "japan_catapillar.md", HierarchicalRetrieverCategory(tagname="土木", level=1)),
    (markdown_dir / "Proffesional_College_of_arts_and_tourism.md", HierarchicalRetrieverCategory(tagname="大学", level=1)),
    (markdown_dir / "students.md", HierarchicalRetrieverCategory(tagname="学生", level=2)),
]

# セッション作成
session = RAGSession(
    vectorstore_dir=vectorstore_dir,
    model_name="gemma3:4b",
    default_template="japanese_concise",
    embedding_name="bge-m3"
)

session.build_vectorstore(file_category_pairs, markdown_dir, overwrite=False)
session.prepare_chain(tagname="土木", level=1)
session.run_interactive(mode="rag")  # ← 従来通り
# session.run_interactive(mode="llm")  # ← RAGを使わず直接生成
