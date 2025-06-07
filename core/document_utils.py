# document_utils.py

import os
import shutil
import tempfile
import re

from typing import Optional, List, Any, Tuple, Dict

import csv
import json

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

# 正規表現：第○章（空白対応・全角数字も対応）
CHAPTER_PATTERN = re.compile(r"^##\s*第\s*[一二三四五六七八九十0-9０-９]+\s*章")

# プレフィックスで階層をさらに下げるべきかを判定
PREFIX_DOWN_PATTERN = re.compile(r"^(###)\s*([ア-ン一二三四五六七八九十0-9０-９]+)")

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

def convert_csv_to_json(csv_file_path: str, json_file_path: str) -> Path:
    """
    指定されたCSVファイルを読み込み、ヘッダーを行のキーとして、
    各行のデータを辞書形式のリストに変換し、指定されたJSONファイルに出力する。

    :param csv_file_path: 読み込むCSVファイルのパス (文字列)
    :param json_file_path: 出力するJSONファイルのパス (文字列)
    :return: 出力されたJSONファイルの Path オブジェクト（絶対パス）
    :raises FileNotFoundError: 入力CSVファイルが見つからない場合
    :raises Exception: ファイル読み書き、CSV解析中にエラーが発生した場合
    """
    data: List[Dict[str, Any]] = []
    csv_path = Path(csv_file_path)

    # 入力ファイル存在チェック
    if not csv_path.exists():
        raise FileNotFoundError(f"入力CSVファイルが見つかりません: {csv_file_path}")

    try:
        # ファイルを読み込みモードで開く
        # encoding='utf-8' を指定
        # newline='' で改行コードの自動変換を防ぐ (CSV読み込みの標準的な推奨設定)
        # errors='replace' でエンコーディングエラーが発生した場合に不正な文字を置き換える
        with open(csv_path, mode='r', encoding='utf-8', newline='', errors='replace') as infile:
            # csv.DictReader を使用すると、最初の行をヘッダーとして扱い、
            # 各行をヘッダーをキーとする辞書として読み込んでくれる
            reader = csv.DictReader(infile)

            # 各行（辞書）をリストに追加
            # DictReaderはヘッダーの数と異なる列を持つ行を自動的に調整しようとしますが、
            # 不正なCSV形式の場合はエラーや予期しない結果になる可能性があります。
            # 提供されたデータ例は末尾にコメント行があるようですが、それらは
            # DictReaderによって無視されるか、例外が発生する可能性があります。
            # JSON例に沿って、ヘッダーに一致する行のみを処理します。
            for row in reader:
                data.append(row)

    except Exception as e:
        # FileNotFoundError は上で捕捉済み
        raise Exception(f"CSVファイルの読み込み中にエラーが発生しました: {e}") from e

    # 出力ファイルのパスをPathオブジェクトに変換
    json_path = Path(json_file_path)

    # 出力ディレクトリが存在しない場合は作成
    # parents=True: 途中のディレクトリもなければ作成
    # exist_ok=True: ディレクトリが既に存在してもエラーにしない
    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # JSONファイルを書き込みモードで開く
        # encoding='utf-8' を指定
        with open(json_path, mode='w', encoding='utf-8') as outfile:
            # 辞書のリストをJSON形式で書き込む
            # ensure_ascii=False で日本語文字列をエスケープせずにそのまま出力（可読性向上）
            # indent=2 で整形して出力 (ネストレベルごとに2つのスペースでインデント)
            json.dump(data, outfile, ensure_ascii=False, indent=2)

    except Exception as e:
        raise Exception(f"JSONファイルの書き込み中にエラーが発生しました: {e}") from e

    # 出力したJSONファイルの Path オブジェクトを絶対パスに解決して返す
    return json_path.resolve()

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

def convert_string_to_markdown(text_string: str, md_path: Path) -> Path | None:
    """
    文字列をMarkdownファイルとして指定されたパスに保存します。

    Parameters:
    - text_string: 保存する文字列
    - md_path: 出力するMarkdownファイルのパス（Path型）

    Returns:
    - 成功時: md_path（Path型）、失敗時: None
    """
    try:
        print(f"文字列を Markdown ファイルとして保存しています: {md_path}")
        # 親ディレクトリが存在しない場合は作成
        md_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(md_path, "w", encoding="utf-8") as fp:
            fp.write(text_string)

        print(f"✅ 文字列を保存しました: {md_path}")
        return md_path

    except Exception as e:
        print(f"文字列の保存中にエラーが発生しました: {e}")
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

    # import json

    # print(f"[DEBUG] セクション数: {len(sections)}")
    # for i, sec in enumerate(sections):
    #     print(f"  - {i+1}: depth={sec['depth']}, name='{sec['name']}', body_len={len(sec['body'])}")

    # print(f"[DEBUG] ネスト構造ルート要素数: {len(root)}")
    # print(json.dumps(root[:1], ensure_ascii=False, indent=2))  # 最初の1要素を構造確認

    return root

def to_half_width_numbers(text: str) -> str:
    return text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

def is_chapter_heading(line: str) -> bool:
    normalized = to_half_width_numbers(line)
    return CHAPTER_PATTERN.match(normalized) is not None

def should_indent_further(line: str) -> bool:
    """カタカナ or 数字で始まるタイトルか？"""
    return bool(PREFIX_DOWN_PATTERN.match(line))

def collect_markdown_headings(md_path: Path) -> list[str]:
    """
    Markdownの'##'見出し行をすべて収集し、階層を調整：
    - 最初の見出しはそのまま
    - 「第○章」が含まれる場合は階層を戻して'##'
    - 通常は'###'
    - ただし、'###' で始まり、かつプレフィックスがカタカナ/数字なら '####'
    """
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    headings = [line.strip() for line in lines if line.strip().startswith("##")]

    adjusted = []
    for i, line in enumerate(headings):
        if i == 0:
            adjusted.append(line)  # 最初の見出しはそのまま
        elif is_chapter_heading(line):
            adjusted.append(re.sub(r"^##", "##", line, count=1))  # 章タイトルはそのまま
        else:
            if should_indent_further(line):  # カタカナ・数字なら1段下げ
                new_line = re.sub(r"^##", "###", line, count=1)
            else:  # 通常は2段下げ
                new_line = re.sub(r"^##", "####", line, count=1)
            adjusted.append(new_line)

    return adjusted

