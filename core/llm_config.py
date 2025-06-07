# from langchain_community.llms.ollama import Ollama as OllamaLLM # type: ignore # この行を削除
from langchain_ollama import OllamaLLM # 新しいインポートパス # type: ignoreは必要に応じて残す
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from . import prompts

def load_llm(model_name: str = "gemma3:4b"):
    """
    指定されたモデル名で LLM（Ollama）を初期化する。
    """
    # インポートしたクラス名に合わせて OllamaLLM を使用
    return OllamaLLM(model=model_name)


def load_prompt(name: str = "japanese_concise") -> ChatPromptTemplate:
    """
    プロンプト名に応じて ChatPromptTemplate を返す。

    Parameters:
    - name: 使用するプロンプト名（例: "japanese_concise", "default", "rephrase"）

    Returns:
    - ChatPromptTemplate: LangChain 互換のプロンプトオブジェクト

    Raises:
    - ValueError: 未定義のプロンプト名が渡された場合
    """
    template_map = {
        "rephrase": prompts.QUESTION_REPHRASE_PROMPT_STR,
        "default": prompts.DEFAULT_COMBINE_PROMPT_STR,
        "japanese_concise": prompts.JAPANESE_CONCISE_PROMPT_STR,
        "english_verbose": prompts.ENGLISH_VERBOSE_PROMPT_STR,
    }

    if name not in template_map:
        raise ValueError(f"未定義のプロンプト名です: '{name}'")

    return ChatPromptTemplate.from_template(template_map[name])



