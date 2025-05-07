import chain_factory

from retriever_utils import RetrieverCategory, HierarchicalRetrieverCategory

from pathlib import Path
from langchain_core.language_models import BaseLanguageModel # type: ignore
from process_utils import process_and_vectorize_file # type: ignore
from interactive_cli import run_rag_cli, run_llm_cli # type: ignore
from llm_config import load_llm, load_prompt # type: ignore

class RAGSession:
    def __init__(
        self,
        vectorstore_dir: Path,
        model_name: str = "gemma3:4b",
        default_template: str = "japanese_concise",
        embedding_name: str = "bge-m3"
    ):
        self.model_name = model_name
        self.prompt_template = load_prompt(default_template)
        self.vectorstore_dir = vectorstore_dir
        self.embedding_name = embedding_name
        self.llm: BaseLanguageModel = load_llm(model_name)
        self.qa_chain = None

    def build_vectorstore(
            self, 
            file_category_pairs: list[tuple[Path, RetrieverCategory]], 
            markdown_dir: Path,
            overwrite: bool):
        for in_file, category in file_category_pairs:
            process_and_vectorize_file(
                in_file=in_file,
                markdown_dir=markdown_dir,
                vectorstore_dir=self.vectorstore_dir,
                category=category,
                embedding_name=self.embedding_name,
                overwrite=overwrite
            )

    def prepare_chain(self, tagname: str, level=int, k: int = 5):
        category = HierarchicalRetrieverCategory(tagname=tagname, level=level)

        self.qa_chain = chain_factory.prepare_chain_for_category(
            llm=self.llm,
            category=category,
            base_path=self.vectorstore_dir,
            chain_type="conversational",
            prompt_template=self.prompt_template,
            k=k,
        )

    def run_interactive(self, mode: str = "rag"):
        """
        対話モード実行

        Parameters:
        - mode: "rag"（RAGチェーン使用）または "llm"（LLM単体使用）

        Raises:
        - RuntimeError: 対応するチェーンまたはモデルが未構築の場合
        """
        if mode == "rag":
            if not self.qa_chain:
                raise RuntimeError("RAGチェーンが構築されていません。prepare_chain() を呼び出してください。")
            run_rag_cli(self.qa_chain)

        elif mode == "llm":
            if not self.llm or not self.prompt_template:
                raise RuntimeError("LLMまたはプロンプトテンプレートが未設定です。")
            run_llm_cli(self.llm, self.prompt_template)

        else:
            raise ValueError(f"未知のモードです: {mode}（'rag' または 'llm' を指定してください）")

