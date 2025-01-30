from transformers import pipeline

coref_pipeline = pipeline("coref-resolution", model="coref-spanbert-large")


def resolve_coreference_in_query(query_text, conversation_history):
    context = "\n".join([f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history])
    full_text = f"{context}\nUser: {query_text}"
    resolved_text = coref_pipeline(full_text)
    resolved_query = resolved_text.split("User:")[-1].strip()
    return resolved_query
