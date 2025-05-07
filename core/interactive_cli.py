def run_rag_cli(qa_chain):
    """
    対話型 CLI：RAG チェーンを利用するバージョン
    """
    print("🔁 RAG 実験モード開始（終了するには 'exit'）")

    while True:
        query = input("\n🗨 質問してください: ")
        if query.strip().lower() == "exit":
            break

        response = qa_chain.invoke({
            "input": query,
            "chat_history": []
        })

        print("\n📄 参照元:")
        for doc in response.get("source_documents", []):
            print("  -", doc.metadata.get("source"))

        print("\n🧠 回答:")
        print(response.get("answer") or response.get("output"))


def run_llm_cli(llm, prompt_template):
    """
    対話型 CLI：単なるLLMとプロンプトのみで直接応答するバージョン
    """
    print("💬 LLM 単体モード開始（終了するには 'exit'）")

    while True:
        query = input("\n🗨 質問してください: ")
        if query.strip().lower() == "exit":
            break

        prompt = prompt_template.format_messages(
            input=query,
            context="（文脈なし）"
        )
        response = llm.invoke(prompt)

        print("\n🧠 回答:")
        print(response)
