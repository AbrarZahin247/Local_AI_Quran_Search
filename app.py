# ==============================================================================
#                      Bangla Semantic Search Engine - app.py
#                (Version with Smart Chunking for Better Context)
# ==============================================================================

import os
import ast
from itertools import groupby
from flask import Flask, render_template, request

# --- Import Core Libraries ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
from rank_bm25 import BM25Okapi

# ==============================================================================
#                      1. ONE-TIME INITIALIZATION
# This entire section runs only once when the application starts.
# ==============================================================================

print("===================================================")
print("Starting Flask server and loading all models...")
print("This may take a moment, please be patient.")
print("===================================================")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
DATA_FILE_PATH = "quran_text.txt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = "faiss_index_bangla"
LLM_MODEL = "gemma:2b"
CHUNK_SIZE_IN_VERSES = 3  # How many verses to group into one document. 3 is a good start.


# --- Load and Process the Source Text File with Smart Chunking ---
print(f"-> Loading data from '{DATA_FILE_PATH}'...")
initial_documents = []
try:
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    # First, parse every line into a preliminary Document object
    for line in raw_lines:
        try:
            line_tuple = ast.literal_eval(line.strip().rstrip(','))
            doc_id, surah_no, ayah_no, text = line_tuple
            initial_documents.append(Document(page_content=text, metadata={'surah': surah_no, 'ayah': ayah_no, 'id': doc_id}))
        except (ValueError, SyntaxError):
            continue # Skip malformed lines
    print(f"--> Successfully loaded {len(initial_documents)} raw verses.")
except FileNotFoundError:
    print(f"!!! FATAL ERROR: Data file not found at '{DATA_FILE_PATH}'.")
    initial_documents = []

# --- Apply Smart Chunking Logic to Group Verses ---
print(f"-> Applying smart chunking to group verses into chunks of {CHUNK_SIZE_IN_VERSES}...")
documents = []
if initial_documents:
    # Group the loaded verses by their Surah number to avoid mixing Surahs in a chunk
    groups = groupby(initial_documents, key=lambda doc: doc.metadata['surah'])
    
    for surah_no, surah_docs_iterator in groups:
        surah_docs = list(surah_docs_iterator)
        for i in range(0, len(surah_docs), CHUNK_SIZE_IN_VERSES):
            # Take a slice of verses (e.g., verses 0-2, then 3-5, etc.)
            chunk_of_docs = surah_docs[i:i + CHUNK_SIZE_IN_VERSES]

            # Combine the text content of the verses in the chunk
            combined_text = " ".join([d.page_content for d in chunk_of_docs])

            # Create a new, more descriptive metadata for the chunk
            start_ayah = chunk_of_docs[0].metadata['ayah']
            end_ayah = chunk_of_docs[-1].metadata['ayah']
            new_meta = {
                'surah': surah_no,
                'ayah_range': f"{start_ayah}-{end_ayah}"
            }

            # Create the new, larger, context-rich document
            new_doc = Document(page_content=combined_text, metadata=new_meta)
            documents.append(new_doc)
            
    print(f"--> Grouped {len(initial_documents)} verses into {len(documents)} context-rich documents.")
else:
    documents = [] # Ensure documents is empty if loading fails


# --- Initialize Embedding Model and Vector Store (FAISS) ---
print(f"-> Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

if os.path.exists(FAISS_INDEX_PATH):
    print(f"-> Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("--> FAISS index loaded successfully.")
else:
    print(f"!!! FAISS index not found. Creating a new one based on smart chunks...")
    print("!!! Please be patient, this one-time setup can take a few minutes.")
    if documents:
        vector_store = FAISS.from_documents(documents, embedding_model)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("--> New FAISS index created and saved successfully.")
    else:
        print("!!! WARNING: No documents loaded, cannot create FAISS index.")
        vector_store = None


# --- Initialize Lexical Search Model (BM25) ---
print("-> Initializing BM25 lexical search index...")
if documents:
    tokenized_corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("--> BM25 index created successfully.")
else:
    print("!!! WARNING: No documents loaded, cannot create BM25 index.")
    bm25 = None


# --- Initialize the Local LLM via Ollama ---
print(f"-> Connecting to local LLM '{LLM_MODEL}' via Ollama...")
try:
    llm = Ollama(model=LLM_MODEL)
    llm.invoke("Hello") 
    print(f"--> Ollama LLM ({LLM_MODEL}) connected and responsive.")
except Exception as e:
    print(f"!!! FATAL ERROR connecting to Ollama: {e}")
    print("!!! Please ensure the Ollama application or 'ollama serve' is running.")
    llm = None

print("===================================================")
print("           Initialization Complete                 ")
print("          Server is ready to accept requests.      ")
print("===================================================")


# ==============================================================================
#                  2. HYBRID SEARCH & EXPLANATION FUNCTION
# ==============================================================================

def perform_hybrid_search_and_explain(query):
    if not vector_store or not bm25:
        return {"query": query, "explanation": "Search system not initialized. Check server logs.", "results": []}
    if not llm:
        return {"query": query, "explanation": "LLM not connected. Please ensure Ollama is running.", "results": []}

    # --- Step A: Perform both searches ---
    vector_results = vector_store.similarity_search_with_score(query, k=10)
    
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_results = [(documents[i], bm25_scores[i]) for i in top_n_bm25_indices]
    
    # --- Step B: Fuse the results using Reciprocal Rank Fusion (RRF) ---
    def reciprocal_rank_fusion(results_lists, k=60):
        fused_scores = {}
        doc_map = {doc.page_content: doc for results in results_lists for doc, _ in results}
        
        for results in results_lists:
            for rank, (doc, score) in enumerate(results):
                doc_content = doc.page_content
                if doc_content not in fused_scores:
                    fused_scores[doc_content] = 0
                fused_scores[doc_content] += 1 / (k + rank + 1)

        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[content] for content, score in reranked_results]

    fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
    
    # --- Step C: Filter out any potential short/bad results after fusion ---
    # This is a final safety net, although the smart chunking should prevent most of it.
    MIN_CHUNK_LENGTH_CHARS = 50 
    filtered_results = [doc for doc in fused_results if len(doc.page_content) > MIN_CHUNK_LENGTH_CHARS]
    
    top_docs = filtered_results  # Select the top 5 fused and filtered results
    # top_docs = filtered_results[:5]  # Select the top 5 fused and filtered results

    # --- Step D: Generate an explanation with the LLM ---
    context = "\n\n".join([f"উৎস: সূরা {doc.metadata['surah']}, আয়াত {doc.metadata['ayah_range']}\nবিষয়বস্তু: {doc.page_content}" for doc in top_docs])
    
    prompt = f"""
    You are a helpful assistant for understanding the Holy Quran in Bengali.
    Based *only* on the following context from the Quran, provide a concise and clear explanation in Bengali for the topic: "{query}"

    Provided Context:
    ---
    {context}
    ---

    Your Explanation in Bengali:
    """
    explanation = llm.invoke(prompt)

    return {
        "query": query,
        "explanation": explanation,
        "results": top_docs
    }


# ==============================================================================
#                  3. FLASK WEB ROUTES
# ==============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    search_results = None
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            print(f"Received search query: '{query}'")
            search_results = perform_hybrid_search_and_explain(query)
    return render_template('index.html', results=search_results)


# ==============================================================================
#                  4. RUN THE FLASK APPLICATION
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)