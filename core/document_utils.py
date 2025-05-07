# document_utils.py

import os
import shutil
import tempfile

from pathlib import Path
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import (  # type: ignore
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader
)

from docling.datamodel.base_models import InputFormat  # type: ignore
from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    SimplePipeline
)  # type: ignore

def get_document_format(file_path: Path) -> InputFormat | None:
    """
    ファイルの拡張子に基づいて、対応する Docling の InputFormat を返す。

    Parameters:
    - file_path: 入力ファイルのパス（Path型）

    Returns:
    - InputFormat または None（未対応形式）
    """
    extension = file_path.suffix.lower()
    format_map = {
        '.docx': InputFormat.DOCX,
        '.pptx': InputFormat.PPTX,
        '.html': InputFormat.HTML,
        '.jpg': InputFormat.IMAGE,
        '.jpeg': InputFormat.IMAGE,
        '.png': InputFormat.IMAGE,
        '.txt': InputFormat.ASCIIDOC,
        '.pdf': InputFormat.PDF,
        '.md': InputFormat.MD,
        '.csv': InputFormat.CSV,
        '.xlsx': InputFormat.XLSX,
        # 将来的な拡張に備えたプレースホルダ
        # '.xml_uspto': InputFormat.XML_USPTO,
        # '.json_docling': InputFormat.JSON_DOCLING,
    }

    return format_map.get(extension, None)

def convert_document_to_markdown(doc_path: Path, md_path: str) -> str | None:
    """
    DOCX や PDF などの入力ファイルを Markdown に変換して保存します。

    Parameters:
    - doc_path: 入力ドキュメントのパス（Path型）
    - md_path: 出力する Markdown ファイルのパス（str型）

    Returns:
    - 成功時: md_path（str型）、失敗時: None
    """
    try:
        input_path = os.path.abspath(str(doc_path))
        print(f"ドキュメントを変換しています: {input_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = os.path.join(temp_dir, os.path.basename(input_path))
            shutil.copy2(input_path, temp_input)

            # パイプラインオプション設定
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True

            converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
                }
            )

            print("変換を開始しています...")
            conv_result = converter.convert(temp_input)
            if not conv_result or not conv_result.document:
                raise ValueError(f"ドキュメントの変換に失敗しました: {doc_path}")

            print("Markdown にエクスポートしています...")
            md = conv_result.document.export_to_markdown()

            with open(md_path, "w", encoding="utf-8") as fp:
                fp.write(md)

            print(f"✅ Markdown を保存しました: {md_path}")
            return md_path

    except Exception as e:
        print(f"ドキュメントの変換中にエラーが発生しました: {e}")
        return None

def suggest_text_splitter(
    doc_path: str,
    documents,
    text_splitter_type: str = "recursive",
    loader_type: str = "markdown"
):
    """
    ドキュメントの構造に基づき適切な chunk_size / overlap を推定して TextSplitter を返す。

    Parameters:
    - doc_path: ファイルパス（主にロギング用）
    - documents: LangChain の Document オブジェクトのリスト
    - text_splitter_type: "recursive" / "character" / "token" など
    - loader_type: "markdown" 指定時は header-based splitter も考慮

    Returns:
    - LangChain TextSplitter インスタンス
    """

    def analyze_document_structure(text: str):
        paragraphs = text.split("\n\n")
        if not paragraphs:
            return 500
        return sum(len(p) for p in paragraphs) / len(paragraphs)

    def suggest_chunk_parameters(text: str, max_context_length=8192):
        avg_len = analyze_document_structure(text)
        if avg_len > 1500:
            chunk_size = min(int(avg_len * 1.2), max_context_length // 2)
            chunk_overlap = int(chunk_size * 0.2)
        elif avg_len > 800:
            chunk_size = 1600
            chunk_overlap = 300
        else:
            chunk_size = 1000
            chunk_overlap = 200
        return chunk_size, chunk_overlap

    sample_text = "\n".join(doc.page_content for doc in documents[:5])
    chunk_size, chunk_overlap = suggest_chunk_parameters(sample_text)
    text_splitter_type = text_splitter_type.lower()

    if text_splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    else:
        raise ValueError(f"Unsupported or unimplemented text_splitter_type: {text_splitter_type}")

def load_documents(doc_path: str, loader_type: str = "markdown"):
    """
    指定されたファイルパスとローダー種別に応じて文書を読み込みます。

    Parameters:
    - doc_path: 入力ファイルパス
    - loader_type: "markdown" / "pdf" / "text" / "csv"

    Returns:
    - LangChain Document オブジェクトのリスト

    Raises:
    - ValueError: サポート外のローダータイプが指定された場合
    """
    loader_type = loader_type.lower()

    if loader_type == "markdown":
        loader = UnstructuredMarkdownLoader(doc_path)
    elif loader_type == "pdf":
        loader = UnstructuredPDFLoader(doc_path)
    elif loader_type == "text":
        loader = TextLoader(doc_path)
    elif loader_type == "csv":
        loader = CSVLoader(doc_path)
    else:
        raise ValueError(f"Unsupported loader_type: {loader_type}")

    return loader.load()


