from Llm.llm_endpoints import get_llm_response


def summarize_conversation(conversation_history):
    try:
        summary_prompt = "Summarize the following conversation:\n" + "\n".join(
            [f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history])
        summary = get_llm_response(summary_prompt)
        print("*************************************************")
        print(summary)
        print("*************************************************")
        return summary
    except:
        return ""
