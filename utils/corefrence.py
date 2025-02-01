import spacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("coreferee")
def resolve_corefrence(query_text, conversation_history):
    combined_text = []
    for turn in conversation_history:
        combined_text.append(f"User:{turn['user']}")
        combined_text.append(f"Bot:{turn['Bot']}")
    combined_text.append(f"User:{query_text}")
    combined_text = "\n".join(combined_text)
    doc = nlp(combined_text)
    resolved_text = doc._.corefrence_resolved
    resolved_query = resolved_text.split('\n')[-1].replace("User: ", "")
    return resolved_query.strip()