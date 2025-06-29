import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, TypedDict
import google.generativeai as genai
import os
import json
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



def process_and_add_new_files(transcripts_folder_path, collection, embedding_model):
    """Process and add new transcript files to the vector database."""
    new_files = get_new_files(transcripts_folder_path, collection)
    if not new_files:
        logging.info("No new files to process")
        return False

    logging.info(f"Found {len(new_files)} new files to process.")

    # Use a reasonable number of workers (4 is usually a good default)
    # Cap workers to avoid overwhelming the system
    n_workers = min(os.cpu_count() or 1, len(new_files), 8) # Use os.cpu_count as a guide, max 8
    logging.info(f"Using {n_workers} workers for processing file reading/splitting.")

    all_chunks = []
    all_metadata = []
    all_ids = []
    processed_count = 0

    # Process files in parallel (reading and splitting)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_single_file, os.path.join(transcripts_folder_path, file)): file
            for file in new_files
        }

        for future in tqdm(as_completed(futures), total=len(new_files), desc="Processing files"):
            file = futures[future]
            try:
                chunks, filename = future.result()
                if not chunks:
                    logging.warning(f"No chunks generated for {filename}, skipping.")
                    continue

                file_metadata = [{"source": filename} for _ in range(len(chunks))]
                # Ensure IDs are unique and valid ChromaDB IDs
                file_ids = [f"{os.path.splitext(filename)[0]}_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadata.extend(file_metadata)
                all_ids.extend(file_ids)

                processed_count += 1
                # logging.info(f"Processed {filename} - {len(chunks)} chunks") # Too verbose in tqdm

            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
                # Continue to the next file


    if not all_chunks:
        logging.info("No valid chunks were processed from new files.")
        return False

    # Process embeddings in batches
    logging.info(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = batch_embed_chunks(all_chunks, embedding_model)

    if len(embeddings) != len(all_chunks):
         logging.error("Mismatch between chunk and embedding count. Aborting upsert.")
         return False

    # Add to database in batches
    batch_size = 500
    logging.info(f"Adding {len(all_chunks)} chunks to database in batches of {batch_size}...")
    try:
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Adding to DB"):
            end_idx = min(i + batch_size, len(all_chunks))
            collection.upsert(
                documents=all_chunks[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=all_metadata[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            # logging.info(f"Added batch {i // batch_size + 1} to database") # Too verbose in tqdm
        logging.info("All batches added.")
    except Exception as e:
        logging.error(f"Error during database upsert: {e}")
        return False


    logging.info(f"Successfully processed and added {processed_count} new files.")
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

def query_router_node(state:GraphState)->dict:
    """
    Determine weather to retrieve  ordocument or generate a general answer
    Args:
        state(GraphState): The current state of the graph.
    Returns:
        dict: The decision("retrieve" or "general")
    """
    logging.info("--Exectuting Query Router")
    logging.info("Before the chane to the state")
    logging.info(f"State on entry:\n {json.dumps(state, indent =2)}")
    query = state['query']
    chat_history = state['chat_history']
    conversation_history = '\n'.join([f"{turn['user']}: {turn['bot']}" for turn in chat_history])

    router_prompt = """
    You are a helpful assistant that routes user queries.
    Based on the user's query and the conversation history, decide if the query
    requires searching a knowledge base about Huberman Lab podcasts (for specific information
    mentioned in transcripts) If the question is regarding any thing realted to neuroscinence 
    or any thing andrew huberman talks in his podcast then you need to use retrieve. Except that
    you need to use general

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
    logging.info("After state has been updated")
    logging.info(f'State on exit : {json.dumps(state, indent=2)}')
    return state


def retrieve_node(state: GraphState, collection, embedding_model) -> GraphState:
    """
    Retrieves documents based on the user query using the provided collection and embedding model.

    Args:
        state (GraphState): The current state of the graph.
        collection: The ChromaDB collection object.
        embedding_model: The sentence transformer embedding model. # Pass model here

    Returns:
        GraphState: The updated state with retrieved_docs and source_links.
    """
    logging.info("---Executing Retrieval---")
    query = state.get('query', '')

    if not query or not collection or not embedding_model:
        logging.error("Retrieval node received invalid state or dependencies.")
        state['retrieved_docs'] = []
        state['source_links'] = []
        state['error'] = state.get('error', '') + "\nRetrieval setup failed."
        return state

    try:
        query_embeddings = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=query_embeddings, n_results=3, include=['documents', 'metadatas'])

        retrieved_docs = results.get('documents', [[]])[0] if results else []
        metadatas = results.get('metadatas', [[]])[0] if results else []

        # Filter out None results if any (Chroma sometimes returns None)
        retrieved_docs = [doc for doc in retrieved_docs if doc is not None]
        metadatas = [meta for meta in metadatas if meta is not None]

        source_links = get_source_link(metadatas) # Use your original function

        state['retrieved_docs'] = retrieved_docs
        state['source_links'] = source_links
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        state['error'] = state.get('error', '') + f"\nError retrieving documents: {e}"
        state['retrieved_docs'] = [] # Ensure keys exist even on error
        state['source_links'] = []

    logging.info(f"Retrieved {len(state.get('retrieved_docs', []))} documents.")
    return state # Return the updated state
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
    
    You are the "General Interaction Handler" for an AI assistant focused on Andrew Huberman's podcast content. Your primary role is to handle simple greetings and answer questions *about* the capabilities and purpose of the overall AI assistant.

    **Your Core Responsibilities:**

    1.  **Handle Greetings:**
        *   Respond politely and appropriately to simple greetings like "Hello", "Hi", "Good morning", "Hey", etc.
        *   Your response should be friendly, professional, and invite further interaction.
        *   Example Input: "Hi" -> Example Output: "Hello! How can I help you today? You can ask me about topics covered in the Huberman Lab podcast or about how this assistant works."
        *   Example Input: "Good morning" -> Example Output: "Good morning! How can I assist you with information from Andrew Huberman's work?"

    2.  **Answer Meta-Questions (Questions about the Bot):**
        *   Respond clearly to questions about the AI assistant's capabilities, function, or purpose. Examples: "What can you do?", "How does this work?", "What is this bot for?".
        *   When explaining capabilities, describe the *overall system's function* (including the RAG part), but make it clear *you* are the part handling this current interaction.
        *   Example Input: "What can you do?" -> Example Output: "I am an AI assistant designed to help you explore content from Andrew Huberman's podcasts. You can ask me specific questions about topics he discusses (like sleep, focus, neuroscience, fitness protocols, etc.). When you ask such a question, I will retrieve relevant information directly from the podcast transcripts and provide you with YouTube links to the specific segments where he discusses it. I can also handle simple greetings like this one!"
        *   Example Input: "How do you work?" -> Example Output: "This assistant uses AI to understand your questions. For specific topics related to Andrew Huberman's content, it retrieves information from a database of his podcast transcripts and identifies relevant YouTube links. For general interactions like greetings or questions about my function, I provide responses like this one."

    **Crucial Constraints (What NOT To Do):**

    *   **DO NOT Attempt to Answer Huberman Content Questions:** You should *not* answer questions like "What does Huberman say about dopamine?", "Tell me about cold plunges", "Explain neuroplasticity". These questions are routed to the specialized RAG node. Your knowledge base does *not* contain the podcast transcripts.
    *   **DO NOT Mention Transcripts or YouTube Links UNLESS Explaining Capabilities:** Only refer to transcript retrieval and YouTube links when specifically asked *what the bot can do* or *how it works*. Do not offer them proactively or in response to greetings.
    *   **DO NOT Impersonate Andrew Huberman:** Maintain a helpful AI assistant persona.
    *   **Keep Responses Concise:** Stick to your designated functions (greetings, meta-questions).

    **Tone:** Helpful, clear, professional, and slightly informative (reflecting the Huberman brand ethos without deep diving into content).

    **Your Goal:** Act as the friendly front-door and information desk for the Huberman AI assistant. Handle simple interactions smoothly and accurately describe the overall service when asked, setting the stage for the RAG node to handle the specific content queries.
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