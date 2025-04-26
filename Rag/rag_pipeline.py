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
from pathlib import Path
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
    # Ensure metadata retrieval is robust
    try:
        existing_metadatas = collection.get(include=['metadatas'])['metadatas']
        existing_files = [meta.get("source") for meta in existing_metadatas if meta and "source" in meta]
    except Exception as e:
        logging.warning(f"Could not retrieve existing metadatas from collection: {e}. Assuming no files processed yet.")
        existing_files = []

    all_files = [f for f in os.listdir(transcripts_folder_path) if f.endswith(".txt")]
    return [f for f in all_files if os.path.basename(f) not in existing_files]


def process_single_file(file_path):
    """Process a single file and return its chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Specify encoding
            content = f.read()
        chunks = split_text_to_chunks(content)
        return chunks, os.path.basename(file_path)
    except Exception as e:
        logging.error(f"Error reading or splitting file {file_path}: {e}")
        return [], os.path.basename(file_path)


def batch_embed_chunks(chunks, embedding_model, batch_size=32):
    """Embed chunks in batches."""
    embeddings = []
    if not chunks: return []
    try:
        # show_progress_bar=False when called from background process maybe?
        for i in tqdm(range(0, len(chunks), batch_size),desc = "Embedding chunks"):
            batch = chunks[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch, show_progress_bar=False) # Set to False for batch process logging clarity
            embeddings.extend(batch_embeddings.tolist())
    except Exception as e:
         logging.error(f"Error during batch embedding: {e}")
         return [] # Return empty on error
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
    conversation_history = '\n'.join([f"{turn['user']}: {turn['bot']}" for turn in chat_history])

    router_prompt = """
    You are a helpful assistant that routes user queries.
    Based on the user's query and the conversation history, decide if the query
    requires searching a knowledge base about Huberman Lab podcasts (for specific information
    mentioned in transcripts) or if it's a general question that can be answered
    without specific document retrieval.

    Respond with ONLY one word: "retrieve" or "general".

    Conversation History:
    {history}

    User Query: {query}
    Decision:
    """.format(history=conversation_history, query=query)
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
    state['decision'] = decision
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
    conversation_history = '\n'.join([f"{turn['user']}: {turn['bot']}" for turn in chat_history])
    general_prompt = """
    You are a helpful assistant. Answer the following question based on your general knowledge.
    Keep the conversation history in mind, but primarily focus on the current query.

    Conversation History:
    {history}

    User Query: {query}

    Answer:
    """.format(history = conversation_history, query=query)
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

def setup_rag_pipeline(chromadb_path :str, transcripts_folder_path:str, collection_name:str="yt_transcripts_collection"):
    """
    Initializes ChromaDB, runs ingestion, and sets up the LangGraph pipeline.

    Args:
        chromadb_path (str): Path to the ChromaDB directory.
        transcripts_folder_path (str): Path to the transcripts folder.
        collection_name (str): Name for the ChromaDB collection.

    Returns:
        tuple: (compiled_langgraph_app, chromadb_collection_object, embedding_model)
               Returns None, None, None if setup fails.
    """
    logging.info("Starting RAG pipeline setup...")
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            logging.info("Google Generative AI configured.")
        else:
            logging.warning("GOOGLE_API_KEY environment variable not set. Generative AI calls may fail.")
        # Initialize ChromaDB client and get collection
        client = chromadb.PersistentClient(path=chromadb_path)

        collection = client.get_or_create_collection(name=collection_name)
        logging.info(f"ChromaDB collection '{collection.name}' ready at {chromadb_path}.")

        # Initialize embedding model
        # This can be resource intensive, do it once
        logging.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Embedding model loaded.")


        # Run the ingestion process
        logging.info("Checking for new transcripts and ingesting...")
        process_and_add_new_files(transcripts_folder_path, collection, embedding_model)
        logging.info("Ingestion process finished.")

        # --- Build the LangGraph ---
        logging.info("Building LangGraph workflow...")
        workflow = StateGraph(GraphState)

        # Add nodes - Pass necessary dependencies (collection, embedding_model)
        workflow.add_node("query_router", query_router_node)
        workflow.add_node("retrieve", lambda state: retrieve_node(state, collection, embedding_model)) # Pass dependencies via lambda
        workflow.add_node("generate_rag_response", generate_rag_response_node)
        workflow.add_node("generate_general_response", generate_general_response_node)
        workflow.add_node("update_history", update_history_node)

        # Set the entry point
        workflow.set_entry_point("query_router")

        # Add edges
        workflow.add_conditional_edges(
            "query_router",
            lambda state: state['decision'], # Logic based on the 'decision' key in state
            {
                "retrieve": "retrieve",
                "general": "generate_general_response",

            }
        )

        workflow.add_edge("retrieve", "generate_rag_response")
        workflow.add_edge("generate_rag_response", "update_history")
        workflow.add_edge("generate_general_response", "update_history")

        # Set the finish point
        workflow.set_finish_point("update_history")

        # Compile the graph
        logging.info("Compiling LangGraph workflow...")
        app = workflow.compile()
        logging.info("LangGraph workflow compiled successfully.")

        return app, collection, embedding_model # Return compiled app and other needed objects

    except Exception as e:
        logging.critical(f"FATAL ERROR during RAG pipeline setup: {e}", exc_info=True)
        return None, None, None 
def run_chat_loop(compiled_app, collection): # collection might not be strictly needed here but can be useful
    """Runs the interactive chat loop using the compiled LangGraph app."""

    if compiled_app is None or collection is None:
        logging.error("RAG pipeline setup failed. Cannot run chat loop.")
        print("Failed to initialize the chatbot. Please check logs.")
        return

    conversation_history = [] # Initialize history for the session

    print("\nChatbot initialized. Type your query or 'exit' to quit.")

    while True:
        query_text = input("\nUser: ").strip()
        if query_text.lower() == "exit":
            print("Ending the conversation. Goodbye!")
            break
        if not query_text:
            print("Please enter a query.")
            continue

        # Prepare the initial state for the graph run
        initial_state: GraphState = {
            'query': query_text,
            'chat_history': conversation_history,
            'decision': '', # This will be set by the router node
            'retrieved_docs': [],
            'source_links': [],
            'final_response': '',
            'error': '' # Initialize error state
        }

        logging.info(f"Invoking graph for query: '{query_text[:50]}...'")
        try:
            # Invoke the compiled graph to process the query
            final_state = compiled_app.invoke(initial_state)

            # Extract results from the final state
            response = final_state.get('final_response', "Sorry, I couldn't generate a response.")
            # Update the external history with the history managed by the graph
            conversation_history = final_state.get('chat_history', conversation_history)
            source_links = final_state.get('source_links', [])
            error = final_state.get('error', '')

            print("-" * 50)
            if source_links:
                 print("Sources:")
                 # Ensure source links are printed clearly
                 for i, link in enumerate(source_links):
                     print(f"- Source {i+1}: {link}")
            print("-" * 50)

            print("\nBot:")
            if error:
                 print(f"An internal process encountered an issue: {error}") # Inform user about potential error
            print(response) # Always print the response, even if there was an error downstream


        except Exception as e:
            logging.error(f"An unhandled error occurred during graph execution: {e}", exc_info=True)
            print(f"Sorry, an unexpected error occurred: {e}")
 

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_root = current_dir.parent 
    default_transcripts_folder = project_root / "Data" / "transcripts"
    default_chromadb_path = current_dir / "chromadb.db" 
    # This runs ingestion and compiles the graph ONCE when the script starts
    rag_app, chroma_collection, embedding_model_instance = setup_rag_pipeline(
        chromadb_path=str(default_chromadb_path),
        transcripts_folder_path=str(default_transcripts_folder)
    )

    if rag_app and chroma_collection:
        run_chat_loop(rag_app, chroma_collection)
    else:
        print("\nChatbot failed to start due to setup errors.")