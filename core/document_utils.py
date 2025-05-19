# document_utils.py

import os
import shutil
import tempfile
import re

from typing import Optional, List, Any, Tuple


from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import (  # type: ignore
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader
)

from docling.datamodel.base_models import InputFormat  # type: ignore
from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
from docling.document_converter import ( # type: ignore
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    SimplePipeline
)

FORMAT_MAP = {
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

DOCUMENT_TYPE= {
    '.docx': "microsoft word",
    '.pptx': "microsoft powerpoint",
    '.html': "html",
    '.jpg': "image",
    '.jpeg': "image",
    '.png': "image",
    '.txt': "plane text",
    '.pdf': "pdf",
    '.md': "markdown",
    '.csv': "csv",
    '.xlsx': "microsoft excel",
    # 将来的な拡張に備えたプレースホルダ
    # '.xml_uspto': InputFormat.XML_USPTO,
    # '.json_docling': InputFormat.JSON_DOCLING,
}

class BaseSplitterStrategy:
    def get_splitter(self, documents: list) -> object:
        raise NotImplementedError("You must implement get_splitter.")

class MarkdownSplitterStrategy(BaseSplitterStrategy):
    def get_splitter(self, documents):
        # ドキュメントの先頭数件からテキストを取得（見出しの検出用サンプル）
        text_sample = "\n".join(doc.page_content for doc in documents[:5])

        header_levels = set()  # 見出しレベル（#の数）を格納する集合

        # テキストを行ごとに確認し、Markdown形式の見出し（#〜######）を検出
        for line in text_sample.splitlines():
            if line.startswith("#"):
                count = len(line) - len(line.lstrip("#"))  # 先頭の#の個数をカウント
                if 1 <= count <= 6:
                    header_levels.add(count)

        # 検出されたレベルを昇順に整列し、LangChain用の形式に変換
        headers = [(f"{'#' * level}", f"header{level}") for level in sorted(header_levels)]

        # 見出しが1つも検出できなかった場合のフォールバック（##レベルのみ）
        if not headers:
            headers = [("##", "header2")]

        # 検出された見出しレベルに基づいて MarkdownHeaderTextSplitter を構築して返す
        from langchain.text_splitter import MarkdownHeaderTextSplitter # type: ignore
        return MarkdownHeaderTextSplitter(headers_to_split_on=headers)

class PlainTextSplitterStrategy(BaseSplitterStrategy):
    def get_splitter(self, documents):
        from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore

        # --- サンプルテキストを生成（先頭5文書の内容を結合） ---
        sample_text = "\n".join(doc.page_content for doc in documents[:5])

        # --- 段落の平均文字数を分析 ---
        def analyze_avg_paragraph_length(text: str):
            paragraphs = text.split("\n\n")  # 空行2つで段落を仮定
            if not paragraphs:
                return 500  # 段落なしの場合のデフォルト値
            total_length = sum(len(p) for p in paragraphs)
            return total_length / len(paragraphs)

        avg_len = analyze_avg_paragraph_length(sample_text)

        # --- 平均長に応じてチャンクサイズとオーバーラップを決定 ---
        if avg_len > 1500:
            chunk_size = 2000
            chunk_overlap = 400
        elif avg_len > 800:
            chunk_size = 1600
            chunk_overlap = 300
        else:
            chunk_size = 1000
            chunk_overlap = 200

        # --- RecursiveCharacterTextSplitter を構築して返す ---
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len  # デフォルトでは len() が文字数ベース
        )

class XMLSplitterStrategy(BaseSplitterStrategy):
    def get_splitter(self, documents):
        raise NotImplementedError("XML対応は今後実装予定です。")

SPLITTER_STRATEGY_MAP = {
    "markdown": MarkdownSplitterStrategy(), # type: ignore
    "text": PlainTextSplitterStrategy(), # type: ignore
    # "pdf": PlainTextSplitterStrategy(),
    # "xml": XMLSplitterStrategy(),
}

def get_document_format(file_path: Path) -> InputFormat | None:
    """
    ファイルの拡張子に基づいて、対応する Docling の InputFormat を返す。

    Parameters:
    - file_path: 入力ファイルのパス（Path型）

    Returns:
    - InputFormat または None（未対応形式）
    """
    extension = Path(file_path).suffix.lower()

    return FORMAT_MAP.get(extension, None)

def get_document_type(file_path: Path) -> str | None:
    extension = Path(file_path).suffix.lower()

    return DOCUMENT_TYPE.get(extension, None)

def convert_document_to_markdown(doc_path: Path, md_path: Path) -> str | None:
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
            return Path(md_path)

    except Exception as e:
        print(f"ドキュメントの変換中にエラーが発生しました: {e}")
        return None

def suggest_text_splitter(documents, loader_type: str = "markdown"):
    """
    与えられた文書とローダー種別に応じて、適切なTextSplitterオブジェクトを返す。
    拡張可能な戦略パターンに基づいて分岐処理を行う。
    """

    # 小文字に正規化（例：Markdown → markdown）
    if not isinstance(loader_type, str):
        raise TypeError(f"loader_type must be str, got {type(loader_type).__name__}")
    loader_type = loader_type.lower()

    # 適切な戦略を取得（無ければPlainText）
    strategy = SPLITTER_STRATEGY_MAP.get(loader_type, PlainTextSplitterStrategy())

    # 戦略に基づいてTextSplitterを生成
    return strategy.get_splitter(documents)

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

def build_nested_structure(md_path: str, max_depth: int = 6) -> list[dict[str, Any]]:
    def split_markdown_nested(md_text: str, max_depth: int = 6) -> list:
        heading_pattern = re.compile(r'^\s*(\#{2,6})\s+(.+)', re.MULTILINE)
        matches = list(heading_pattern.finditer(md_text))

        sections = []
        order = 0

        for i, match in enumerate(matches):
            depth = len(match.group(1))
            if depth > max_depth:
                continue
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)

            next_same_or_higher_depth_pos = None
            for j in range(i + 1, len(matches)):
                if len(matches[j].group(1)) <= depth:
                    next_same_or_higher_depth_pos = matches[j].start()
                    break
            if next_same_or_higher_depth_pos:
                end = next_same_or_higher_depth_pos

            body = md_text[start:end].strip()
            order += 1
            sections.append({
                "order": order,
                "depth": depth,
                "name": title,
                "body": body
            })

        return sections

    root = []
    stack = []

    md_text = Path(md_path).read_text(encoding="utf-8") 
    sections = split_markdown_nested(md_text, max_depth=max_depth)

    for section in sections:
        node = {
            "order": section["order"],
            "depth": section["depth"],
            "name": section["name"],
            "body": section["body"],
            "children": []
        }

        while stack and stack[-1]["depth"] >= node["depth"]:
            stack.pop()

        if not stack:
            root.append(node)
        else:
            stack[-1]["children"].append(node)

        stack.append(node)

    return root



