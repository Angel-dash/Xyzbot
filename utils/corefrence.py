import spacy
from spacy.tokens import Doc
import coreferee

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("coreferee")

# Register the custom extension attribute
Doc.set_extension('resolved_text', default=None, force=True)


def resolve_coreferences(query_text, conversation_history):
    """
    Resolve coreferences in the given text using spaCy and coreferee.

    Args:
        query_text (str): The current query to resolve
        conversation_history (list): List of dictionaries containing previous conversation turns

    Returns:
        str: Text with resolved coreferences
    """
    # Combine conversation history and current query
    combined_text = []
    for turn in conversation_history:
        combined_text.append(f"User: {turn['user']}")
        combined_text.append(f"Bot: {turn['Bot']}")
    combined_text.append(f"User: {query_text}")
    text = "\n".join(combined_text)

    # Process the text
    doc = nlp(text)

    # Get all tokens and their potential antecedents
    resolved_tokens = list(doc)

    # Resolve coreferences
    for chain in doc._.coref_chains:
        for mention in chain:
            if mention.root_index != chain.most_specific.root_index:
                # Replace mention with its antecedent
                resolved_tokens[mention.root_index] = doc[chain.most_specific.root_index]

    # Reconstruct the text with resolved references
    resolved_text = "".join([token.text_with_ws if isinstance(token, spacy.tokens.Token)
                             else token.text + " " for token in resolved_tokens])

    # Extract the resolved query (last line)
    resolved_query = resolved_text.split('\n')[-1].replace("User: ", "").strip()

    return resolved_query