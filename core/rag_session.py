from pathlib import Path
from langchain_core.language_models import BaseLanguageModel # type: ignore

from . import chain_factory
from . import llm_config
from . import retriever_utils
from . import process_utils
from . import ingestion

class RAGSession:
    class VectorStoreEntry:
        def __init__(self, file_path_str: str, category: retriever_utils.RetrieverCategory):
            self.file_path = Path(file_path_str)
            self.category = category

        def __repr__(self):
            return f"VectorStoreEntry(path={self.file_path}, category={self.category})"
        
    def __init__(
        self,
        model_name: str = None,
        default_template: str = None,
        embedding_name: str = None
    ):
        self.model_name = model_name
        self.prompt_template = llm_config.load_prompt(default_template)
        self.embedding_name = embedding_name
        self.llm: BaseLanguageModel = llm_config.load_llm(model_name)
        self.qa_chain = None

        print(f"DEBUG RAGSession: Loaded prompt template string: {self.prompt_template}")
        if self.prompt_template is None:
            print("DEBUG RAGSession: Failed to load prompt template!")

    def build_vectorstore(
            self,
            entries: list[VectorStoreEntry],
            markdown_dir: Path,
            overwrite: bool):
        """
        各 VectorStoreEntry に基づいて、ベクトルストアを構築する。
        :param entries: VectorStoreEntryのリスト（file_path と category を保持）
        :param markdown_dir: Markdown変換後の格納ディレクトリ
        :param overwrite: 既存のベクトルストアを上書きするかどうか
        """
        for entry in entries:
            process_utils.process_and_vectorize_file(
                in_file=entry.file_path,
                markdown_dir=markdown_dir,
                category=entry.category,
                embedding_name=self.embedding_name,
                overwrite=overwrite
            )

    def prepare_chain(
        self,
        vectorstore_dir,
        category: retriever_utils.RetrieverCategory,
        k: int = 5
    ):
        """
        RetrieverCategory の種類に応じて、適切なチェーンを構築する。
        """

        if isinstance(category, retriever_utils.HierarchicalRetrieverCategory):
            # 階層型カテゴリに対する処理
            self.qa_chain = chain_factory.prepare_chain_for_category(
                llm=self.llm,
                category=category,
                base_path=vectorstore_dir,
                chain_type="conversational",
                prompt_template=self.prompt_template,
                k=k,
            )

        elif isinstance(category, retriever_utils.FlatRetrieverCategory):
            # フラット型カテゴリに対する処理（例）
            self.qa_chain = chain_factory.prepare_flat_chain(
                llm=self.llm,
                category=category,
                base_path=vectorstore_dir,
                prompt_template=self.prompt_template,
                k=k,
            )

        else:
            raise TypeError(f"Unsupported category type: {type(category).__name__}")

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
            
            print("🔁 RAG 実験モード開始（終了するには 'exit'）")
            while True:
                query = input("\n🗨 質問してください: ")
                if query.strip().lower() == "exit":
                    break

                response = self.qa_chain.invoke({
                    "input": query,
                    "chat_history": []
                })

                print("\n📄 参照元:")
                for doc in response.get("source_documents", []):
                    print("  -", doc.metadata.get("source"))

                print("\n🧠 回答:")
                print(response.get("answer") or response.get("output"))

        elif mode == "llm":
            if not self.llm or not self.prompt_template:
                raise RuntimeError("LLMまたはプロンプトテンプレートが未設定です。")
            
            print("💬 LLM 単体モード開始（終了するには 'exit'）")
            while True:
                query = input("\n🗨 質問してください: ")
                if query.strip().lower() == "exit":
                    break

                prompt = self.prompt_template.format_messages(
                    input=query,
                    context="（文脈なし）"
                )
                response = self.llm.invoke(prompt)

                print("\n🧠 回答:")
                print(response)

        else:
            raise ValueError(f"未知のモードです: {mode}（'rag' または 'llm' を指定してください）")

