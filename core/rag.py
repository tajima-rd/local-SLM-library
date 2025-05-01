# 必要なパッケージをインストールします
import os, tempfile, shutil, json
from pathlib import Path 
from uuid import uuid4 


# Docling のインポート
from docling.datamodel.base_models import InputFormat  # type: ignore
from docling.datamodel.pipeline_options import PdfPipelineOptions,TesseractCliOcrOptions  # type: ignore
from docling.document_converter import DocumentConverter,PdfFormatOption,WordFormatOption,SimplePipeline  # type: ignore

# LangChain imports 
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_ollama import OllamaEmbeddings,OllamaLLM  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain.chains import ConversationalRetrievalChain  # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore

def suggest_text_splitter(
    doc_path: str,
    documents,
    text_splitter_type: str = "recursive",
    loader_type: str = "markdown"
):
    """ドキュメントと形式に基づいて TextSplitter を提案する"""

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

    # 文書の一部からテキストをサンプリング
    sample_text = "\n".join(doc.page_content for doc in documents[:5])
    chunk_size, chunk_overlap = suggest_chunk_parameters(sample_text)

    # スプリッター選択
    text_splitter_type = text_splitter_type.lower()

    if text_splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    elif text_splitter_type == "character":
        return CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif text_splitter_type == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif text_splitter_type == "markdown":
        if loader_type != "markdown":
            raise ValueError("MarkdownHeaderTextSplitter can only be used with 'markdown' loader_type.")
        return MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
    else:
        raise ValueError(f"Unsupported text_splitter_type: {text_splitter_type}")


def get_document_format(file_path) -> InputFormat: 
    """ファイル拡張子に基づいてドキュメント形式を決定します""" 
    try: 
        file_path = str(file_path) 
        extension = os.path.splitext(file_path)[1].lower() 
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
            '.xlsx': InputFormat.XLSX
            # '.xml_uspto': InputFormat.XML_USPTO,
            # '.xml_jats': InputFormat.XML_JATS,
            # '.json_docling': InputFormat.JSON_DOCLING,
        } 
        return format_map.get(extension, None) 
    except: 
        return "get_document_format でエラーが発生しました: {str(e)}"

def convert_document_to_markdown(doc_path, md_path):
    """簡素化されたパイプラインを使用してドキュメントをマークダウンに変換します""" 
    try: 
        # 絶対パスに変換します string 
        input_path = os.path.abspath(str(doc_path)) 
        print(f"ドキュメントを変換しています: {doc_path}") 

        # 処理用の一時ディレクトリを作成します
        with tempfile.TemporaryDirectory() as temp_dir: 
            # 入力ファイルを一時ディレクトリにコピーします
            temp_input = os.path.join(temp_dir, os.path.basename(input_path)) 
            shutil.copy2(input_path, temp_input) 

            # パイプライン オプションを構成します
            pipeline_options = PdfPipelineOptions() 
            pipeline_options.do_ocr = False # OCR を一時的に無効にします
            pipeline_options.do_table_structure = True 

            # 最小限のオプションでコンバーターを作成します
            converter = DocumentConverter( 
                allowed_formats=[ 
                    InputFormat.PDF, 
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                ],
                format_options={ 
                    InputFormat.PDF: PdfFormatOption( 
                        pipeline_options=pipeline_options, 
                    ), 
                    InputFormat.DOCX: WordFormatOption( 
                        pipeline_cls=SimplePipeline 
                    ) 
                } 
            ) 
            # ドキュメントを変換する
            print("変換を開始しています...") 
            conv_result = converter.convert(temp_input) 
            if not conv_result or not conv_result.document: 
                raise ValueError(f"ドキュメントの変換に失敗しました: {doc_path}")
            
            # markdown にエクスポートする
            print("markdown にエクスポートしています...") 
            md = conv_result.document.export_to_markdown()

            # マークダウンファイルを書き込む
            print(f"マークダウンを次の場所に書き込んでいます: {md_path}") 

            with open(md_path, "w", encoding="utf-8") as fp: 
                fp.write(md) 
            return md_path 
    except:
        return f"ドキュメントの変換中にエラーが発生しました: {doc_path}"

def prepare_markdown(in_file: str | Path, md_path: str | Path) -> Path:
    """
    入力ファイルを Markdown に変換し、md_path に保存する。
    ただし、入力ファイルがすでに Markdown の場合はそのまま返す。

    Parameters:
        in_file: 入力ファイルのパス（PDF, Word, Markdown など）
        md_path: Markdown 形式ファイルの出力先パス

    Returns:
        Path: 利用すべき Markdown ファイルのパス
    """
    doc_path = Path(in_file)
    doc_format = get_document_format(doc_path)

    if not doc_format:
        raise ValueError(f"Unsupported document format: {doc_path.suffix}")

    if doc_format == InputFormat.MD:
        return doc_path  # 変換不要、オリジナルをそのまま返す

    md_path = Path(md_path)

    if not md_path.exists():
        convert_document_to_markdown(doc_path, md_path)

    return md_path


def save_chain(
        doc_path,
        vect_path,
        embeddings,
        category,
        loader_type="markdown",
        text_splitter_type="recursive"
    ):
    """ドキュメントを処理し、ベクトルストアを作成・保存する。成功ならTrue、失敗ならFalseを返す。"""
    try:
        # ローダーを選択
        loader_type = loader_type.lower()
        if loader_type == "markdown":
            loader = UnstructuredMarkdownLoader(str(doc_path))
        elif loader_type == "pdf":
            loader = UnstructuredPDFLoader(str(doc_path))
        elif loader_type == "text":
            loader = TextLoader(str(doc_path))
        elif loader_type == "csv":
            loader = CSVLoader(str(doc_path))
        else:
            raise ValueError(f"Unsupported loader_type: {loader_type}")

        documents = loader.load()

        # テキスト分割
        splitter = suggest_text_splitter(
            doc_path=doc_path,
            documents=documents,
            text_splitter_type=text_splitter_type,
            loader_type=loader_type
        )
        texts = splitter.split_documents(documents)

        # ✅ 各文書に一意のIDを追加
        for doc in texts:
            doc.metadata["doc_id"] = str(uuid4())

        # ベクトルストア作成と保存
        vectorstore = FAISS.from_documents(texts, embeddings.model)
        vectorstore.save_local(vect_path)

        # メタデータ保存
        metadata = {
            "embedding_model": embeddings.name,
            "loader_type": loader_type,
            "text_splitter_type": text_splitter_type,
            "category":category,
        }
        with open(os.path.join(vect_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        return True  # 成功
    except Exception as e:
        print(f"ベクトルストア保存中にエラーが発生しました: {e}")
        return False  # 失敗

def load_vectorstore(vect_path):
    """ベクトルストア (FAISS) をロードする。成功すればVectorstore、失敗すればNoneを返す。"""
    try:
        with open(os.path.join(vect_path, "metadata.json")) as f:
            metadata = json.load(f)
        
        embeddings = OllamaEmbeddings(model=metadata["embedding_model"])
        vectorstore = FAISS.load_local(vect_path, embeddings, allow_dangerous_deserialization=True)
        
        return vectorstore
    except Exception as e:
        print(f"ベクトルストアのロードに失敗しました: {e}")
        return None

def create_retriever(vectorstore, k: int = 5, score_threshold: float = None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    if score_threshold is not None:
        retriever.score_threshold = score_threshold  # 仮想プロパティ例
    return retriever

def load_retriever(vect_path):
    """ベクトルストアをロードし、Retrieverとして返す。失敗すればNone。"""
    vectorstore = load_vectorstore(vect_path)
    if vectorstore is not None:
        return vectorstore.as_retriever()
    else:
        return None

def merge_vectorstore(vect_paths):
    """
    複数のFAISSベクトルストアを再構築して統合する。
    - merge_from() を使用せず、from_documents で明示的に再構築する。
    - 各ベクトルストアの metadata.json から embedding_model を取得して一致を検証。
    - 一致しない場合は ValueError。
    Returns:
        FAISS vectorstore object
    """
    print("[DEBUG] 統合対象パス:")
    for p in vect_paths:
        print("  -", p)
    if not vect_paths:

        raise ValueError("ベクトルストアパスリストが空です。")

    all_docs = []
    embedding_models = set()

    for path in vect_paths:
        # ベクトルストアを読み込む（Noneならスキップ）
        store = load_vectorstore(path)
        if store is None:
            print(f"⚠️ ベクトルストアの読み込みに失敗: {path}")
            continue

        # 文書を取得し、UUIDを割り当てて追加
        docs = list(store.docstore._dict.values())
        for doc in docs:
            doc.metadata["doc_id"] = str(uuid4())
        all_docs.extend(docs)

        # メタデータからembeddingモデル名を取得
        metadata_path = os.path.join(path, "metadata.json")
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                embedding_models.add(metadata["embedding_model"])
        except Exception as e:
            print(f"⚠️ メタデータの読み込みに失敗: {metadata_path}\n{e}")
            continue

    if not all_docs:
        raise RuntimeError("統合対象の文書が存在しません。")

    if len(embedding_models) != 1:
        raise ValueError(f"複数の埋め込みモデルが混在しています: {embedding_models}")

    # 一致したembeddingモデルを使って再構築
    embedding_model = embedding_models.pop()
    embeddings = OllamaEmbeddings(model=embedding_model)

    print(f"✅ {len(all_docs)} 件の文書を使用してベクトルストアを再構築します（モデル: {embedding_model}）")
    merged_store = FAISS.from_documents(all_docs, embeddings)
    return merged_store


def edit_vectorstore_metadata(vectorstore, edit_function):
    """
    ベクトルストア内のすべてのメタデータを編集する関数

    Args:
        vectorstore (FAISS): 編集対象のベクトルストア
        edit_function (Callable): 引数にDocumentを受け取り、変更後のmetadataを返す関数

    Returns:
        FAISS: メタデータが更新されたベクトルストア
    """
    # 全Documentを取得
    docs = vectorstore.docstore._dict.values()

    # メタデータを更新
    for doc in docs:
        new_metadata = edit_function(doc)
        if isinstance(new_metadata, dict):
            doc.metadata = new_metadata
        else:
            raise ValueError("edit_functionは新しいmetadata辞書を返す必要があります")

    return vectorstore






