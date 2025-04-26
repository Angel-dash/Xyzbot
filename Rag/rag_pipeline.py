import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, TypedDict
import google.generativeai as genai
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from Llm.llm_endpoints import get_llm_response
from utils.get_link import get_source_link
from Prompts.huberman_prompt import huberman_prompt
from tqdm import tqdm
from langgraph.graph import StateGraph, END
# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

chromadb_path = "app/Rag/chromadb.db"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Helper Functions
def split_text_to_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(docs)
    return chunks


def get_new_files(transcripts_folder_path, collection):
    """Find new transcript files that haven't been processed yet."""
    all_files = [f for f in os.listdir(transcripts_folder_path) if f.endswith(".txt")]
    existing_files = [meta["source"] for meta in collection.get()['metadatas']]
    return [f for f in all_files if f not in existing_files]


def process_single_file(file_path):
    """Process a single file and return its chunks."""
    with open(file_path, 'r') as f:
        content = f.read()
    chunks = split_text_to_chunks(content)
    return chunks, os.path.basename(file_path)


def batch_embed_chunks(chunks, batch_size=32):
    """Embed chunks in batches."""
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size),desc = "Embedding chunks"):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_embeddings.tolist())
    return embeddings


def process_and_add_new_files(transcripts_folder_path, collection):
    """Process and add new transcript files to the vector database."""
    new_files = get_new_files(transcripts_folder_path, collection)
    if not new_files:
        logging.info("No new files to process")
        return False

    # Use a reasonable number of workers (4 is usually a good default)
    n_workers = min(8, len(new_files))
    logging.info(f"Using {n_workers} workers for processing")

    all_chunks = []
    all_metadata = []
    all_ids = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_single_file, os.path.join(transcripts_folder_path, file)): file
            for file in new_files
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                chunks, filename = future.result()
                file_metadata = [{"source": filename} for _ in range(len(chunks))]
                file_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadata.extend(file_metadata)
                all_ids.extend(file_ids)

                logging.info(f"Processed {filename}")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
                continue

    # Process embeddings in batches
    logging.info(f"Generating embeddings for {len(all_chunks)} chunks")
    embeddings = batch_embed_chunks(all_chunks)

    # Add to database in batches
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        collection.upsert(
            documents=all_chunks[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=all_metadata[i:end_idx],
            ids=all_ids[i:end_idx]
        )
        logging.info(f"Added batch {i // batch_size + 1} to database")

    logging.info(f"Successfully processed {len(new_files)} files")
    return True


def query_database(collection, query_text, n_results=3):
    """Retrieve the most relevant chunks for the query."""
    query_embeddings = embedding_model.encode(query_text).tolist()
    results = collection.query(query_embeddings=query_embeddings, n_results=n_results)
    retrieved_docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    return retrieved_docs, metadatas


def enhance_query_with_history(query_text, summarized_history):
    enhance_query = f"{query_text}*2\n\n{summarized_history}"
    return enhance_query


def update_conversation_history(history, user_query, bot_response):
    """Update and keeps track of conversation history between user and the bot."""
    history.append({"user": user_query, "bot": bot_response})
    return history


def generate_response(conversation_history, query_text, retrieved_docs, source_links):
    """Generate a response using retrieved documents and the generative AI model."""
    context = " ".join(retrieved_docs)
    history_str = "\n".join([f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history])
    sources_str = "\n".join(source_links)

    prompt = huberman_prompt.format(
        context=context,
        sources=sources_str,
        history=history_str,
        question=query_text
    )

    response = get_llm_response(prompt)
    full_response = f"{response}\n\nSources:\n{sources_str}"
    return full_response

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The user's query string.
        chat_history: List of previous conversation turns (user/bot).
        decision: The routing decision ("retrieve" or "general").
        retrieved_docs: List of retrieved document chunks.
        source_links: List of source URLs/identifiers.
        final_response: The final generated response string.
        error: Any error message encountered.
    """
    query: str
    chat_history: List[Dict[str, str]]
    decision: str
    retrieved_docs: List[str]
    source_links: List[str]
    final_response: str
    error: str 

def query_router_node(state:GraphState)->str:
    """
    Determine weather to retrieve  ordocument or generate a general answer
    Args:
        state(GraphState): The current state of the graph.
    Returns:
        str: The decision("retrieve" or "general")
    """
    logging.info("--Exectuting Query Router")
    query = state['query']
    chat_history = state['chat_history']


    router_prompt = f"""
    You are a helpful assistant that routes user queries.
    Based on the user's query and the conversation history, decide if the query
    requires searching a knowledge base about Huberman Lab podcasts (for specific information
    mentioned in transcripts) or if it's a general question that can be answered
    without specific document retrieval.

    Respond with ONLY one word: "retrieve" or "general".

    Conversation History:
    {'\n'.join([f"{turn['user']}: {turn['bot']}" for turn in chat_history])}

    User Query: {query}

    Decision:
    """
    try: 
        decision = get_llm_response(router_prompt).strip().lower()
        if decision not in ["retrieve", "general"]:
            logging.warning(f"Router return unexpected output: {decision}. Defaulting to general")
            decision = 'general'
    except Exception as e:
        logging.error(f"Error during query routing {e}")
        decision = "general"
        state['error'] = f"Error routing query {e}"
    logging.info(f"Router decision: {decision}")
    state['decision'] #Update the state with decision 
    return decision


def retrieve_node(state:GraphState, collection)->GraphState:
    """
    Retrives node based on user query. 
    Args:
        state(GraphState): The current state of the graph. 
        collection: The ChromaDb collection object
    
    Returns:
        GraphState: The updated state with retirved_docs and source_links
    """
    logging.info("--Executing Retrivel--")
    query = state['query']
    try:
        retrieved_docs, metadatas= query_database(collection, query)
        source_links = get_source_link(metadatas)
        state['retrieved_docs'] = retrieved_docs
        state['source_links'] = source_links
    except Exception as e: 
        logging.error(f"Error during retrivel {e}")
        state['error'] = f"Error retriveing doc: {e}"
        state['retrieved_docs'] = []
        state['source_links'] = []
    logging.info(f"Retrieved {len(state.get('retrieved_docs', []))} documents.")
    return state

def generate_rag_response_node(state:GraphState)->GraphState:
    """
    Generate a response using retrived documents as context
    Args:
        state (GrapState): The current state of the graph. 
    Returns: 
        GraphState: The updated state with final response
    """
    logging.info("--Generatated Rag Response")
    query = state['query']
    chat_history = state["chat_history"]
    retrieved_docs = state['retrieved_docs']
    source_links = state['source_links']
    if not retrieved_docs:
        logging.warning("No documents retrieved for RAG response.")
        # Handle case where retrieval failed or found nothing
        state['final_response'] = "I couldn't find relevant information in my knowledge base for that. Can I help with something else?"
        return state # Return state early

    try:
        # Call your original function, adapted to take state components
        response = generate_response(chat_history, query, retrieved_docs, source_links)
        state['final_response'] = response
    except Exception as e:
        logging.error(f"Error generating RAG response: {e}")
        state['error'] = f"Error generating response: {e}"
        state['final_response'] = "Sorry, I encountered an error while trying to generate a response."


    logging.info("RAG Response generated.")
    return state

def generate_general_response_node(state:GraphState)-> GraphState:
    """
    Generates a response for general questions without retrieval.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        GraphState: The updated state with the final_response.
    """
    logging.info("---Generating General Response---")
    query = state['query']
    chat_history = state['chat_history']
    general_prompt = f"""
    You are a helpful assistant. Answer the following question based on your general knowledge.
    Keep the conversation history in mind, but primarily focus on the current query.

    Conversation History:
    {'\n'.join([f"{turn['user']}: {turn['bot']}" for turn in chat_history])}

    User Query: {query}

    Answer:
    """
    try:
        response = get_llm_response(general_prompt)
        state['final_response'] = response 
        state['source_links']= []

    except Exception as e:
        logging.error(f"Error generating general response: {e}")
        state['error'] = f"Error generating general response: {e}"
        state['final_response'] = "Sorry, I encountered an error while trying to answer that general question."

    logging.info("General Response generated.")
    return state 

def update_history_node(state:GraphState)-> GraphState:
    """
    Update the conversation history
    Args: 
        state(GraphState): the current state of the graph 
    Returns:
        GraphState: The updated state with chat history
    """
    logging.info("---Updating History---")
    query = state['query']
    response = state['final_response']
    chat_history = state['chat_history'] 
    chat_history = update_conversation_history(chat_history, query, response)
    state['chat_history'] = chat_history
    logging.info("History updated.")
    return state

workflow = StateGraph(GraphState)
workflow.add_node("query_router", query_router_node)
workflow.add_node("retrieve", lambda state: retrieve_node(state, collection)) 
workflow.add_node("generate_rag_response", generate_rag_response_node)
workflow.add_node("generate_general_response", generate_general_response_node)
workflow.add_node("update_history", update_history_node)

# Set the entry point
workflow.set_entry_point("query_router")

#Conditional edge cases
workflow.add_conditional_edges(
    "query_router",
    # The function to call to decide the next node
    lambda state: state['decision'],
    {
        "retrieve": "retrieve", # If state['decision'] is 'retrieve', go to 'retrieve' node
        "general": "generate_general_response", # If state['decision'] is 'general', go to 'generate_general_response' node
    }
)
workflow.add_edge("retrieve", "generate_rag_response")
workflow.add_edge("generate_rag_response", "update_history")
workflow.add_edge("generate_general_response", "update_history")
workflow.set_finish_point("update_history")
app = workflow.compile()
def main_workflow_with_langgraph(transcripts_folder_path, collection):
    """Run the full RAG workflow."""
    new_files_added = process_and_add_new_files(transcripts_folder_path, collection)
    if new_files_added:
        logging.info("New transcripts added to the database.")
    else:
        logging.info("No new files found. Using existing database.")

    conversation_history = []

    while True:
        query_text = input("\nEnter your query(or type 'exit' to end):").strip()
        if query_text.lower() == "exit":
            print("Ending the conversation. Goodbye")
            break
        initial_state: GraphState ={
            'query':query_text, 
            'chat_history':conversation_history,
            'decision':'',
            'retrieved_docs': [],
            'source_links': [],
            'final_response': '',
            'error': ''

        }
        logging.info(f"Starting graph run for query: {query_text}")
        try:
            # Invoke the compiled graph
            final_state = app.invoke(initial_state)

            # Extract results from the final state
            response = final_state['final_response']
            conversation_history = final_state['chat_history'] # Get the updated history
            source_links = final_state['source_links'] # Get source links if any
            error = final_state['error']

            print("-" * 50)
            if source_links:
                 print("Sources:")
                 for link in source_links:
                     print(link)
            print("-" * 50)


            print("\nGenerated Response:")
            if error:
                 print(f"An error occurred: {error}")
                 print(response) # Print the potentially partial/error response
            else:
                 print(response)

        except Exception as e:
            logging.error(f"An unhandled error occurred during graph execution: {e}")
            print(f"Sorry, an internal error occurred: {e}")
