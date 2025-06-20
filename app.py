# ==============================================================================
#                      Bangla Semantic Search Engine - app.py
#                (Version with Hybrid and Exact Match Search Modes)
# ==============================================================================
import os
import sys
import ast
import math
from itertools import groupby
from flask import Flask, render_template, request

# --- Import Core Libraries (Unchanged) ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi

# ==============================================================================
#                      1. ONE-TIME INITIALIZATION (Unchanged)
# ==============================================================================
# (This entire section is the same as the previous version. It loads the models
# and processes the text file into document chunks for BM25)
print("===================================================")
print("Starting Flask server and loading pre-built models...")
# ... [same initialization code as before] ...
# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
DATA_FILE_PATH = "quran_text.txt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = "faiss_index_bangla"
CHUNK_SIZE_IN_VERSES = 3

# --- Load documents for BM25 search (Unchanged) ---
print(f"-> Loading document text for keyword search...")
try:
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    initial_documents = []
    for line in raw_lines:
        try:
            line_tuple = ast.literal_eval(line.strip().rstrip(','))
            doc_id, surah_no, ayah_no, text = line_tuple
            initial_documents.append(Document(page_content=text, metadata={'surah': surah_no, 'ayah': ayah_no, 'id': doc_id}))
        except (ValueError, SyntaxError): continue
    
    documents = []
    groups = groupby(initial_documents, key=lambda doc: doc.metadata['surah'])
    for surah_no, surah_docs_iterator in groups:
        surah_docs = list(surah_docs_iterator)
        for i in range(0, len(surah_docs), CHUNK_SIZE_IN_VERSES):
            chunk_of_docs = surah_docs[i:i + CHUNK_SIZE_IN_VERSES]
            combined_text = " ".join([d.page_content for d in chunk_of_docs])
            start_ayah, end_ayah = chunk_of_docs[0].metadata['ayah'], chunk_of_docs[-1].metadata['ayah']
            new_meta = {'surah': surah_no, 'ayah_range': f"{start_ayah}-{end_ayah}"}
            documents.append(Document(page_content=combined_text, metadata=new_meta))
    print(f"--> Loaded {len(documents)} document chunks for BM25.")
except FileNotFoundError:
    print(f"!!! FATAL ERROR: Data file '{DATA_FILE_PATH}' not found.")
    documents = []

# --- Load the PRE-BUILT Search Models (Unchanged) ---
print(f"-> Loading embedding model and FAISS index...")
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
if not os.path.exists(FAISS_INDEX_PATH):
    print(f"!!! FATAL ERROR: FAISS index not found at '{FAISS_INDEX_PATH}'.")
    print(f"!!! Please run the 'create_index.py' script first to generate the index.")
    sys.exit(1)
vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
print("--> FAISS index loaded successfully.")

print("-> Initializing BM25 lexical search index...")
if documents:
    tokenized_corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("--> BM25 index created successfully.")
else: bm25 = None

print("===================================================")
print("           Initialization Complete                 ")
# ==============================================================================


# ==============================================================================
#                  2. SEARCH FUNCTIONS (UPDATED)
# ==============================================================================

def perform_hybrid_search(query):
    # This function remains the same as the previous version
    if not vector_store or not bm25: return []
    vector_results = vector_store.similarity_search_with_score(query, k=500)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:500]
    bm25_results = [(documents[i], bm25_scores[i]) for i in top_n_bm25_indices]
    
    def reciprocal_rank_fusion(results_lists, k=60):
        fused_scores, doc_map = {}, {doc.page_content: doc for results in results_lists for doc, _ in results}
        for results in results_lists:
            for rank, (doc, score) in enumerate(results):
                doc_content = doc.page_content
                if doc_content not in fused_scores: fused_scores[doc_content] = 0
                fused_scores[doc_content] += 1 / (k + rank + 1)
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[content] for content, score in reranked_results]
    
    fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
    MIN_CHUNK_LENGTH_CHARS = 50 
    return [doc for doc in fused_results if len(doc.page_content) > MIN_CHUNK_LENGTH_CHARS]

# ** NEW FUNCTION FOR EXACT MATCH SEARCH **
def perform_exact_search(query):
    """
    Performs a keyword-only (BM25) search and returns results that contain
    the keywords.
    """
    if not bm25: return []
    
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get all documents with a score greater than 0 (meaning at least one keyword matched)
    matching_docs_with_scores = []
    for i, score in enumerate(bm25_scores):
        if score > 0:
            matching_docs_with_scores.append((documents[i], score))
    
    # Sort the matching documents by their score in descending order
    sorted_exact_results = sorted(matching_docs_with_scores, key=lambda item: item[1], reverse=True)
    
    # Extract just the document objects from the sorted list
    all_relevant_docs = [doc for doc, score in sorted_exact_results]
    
    # Filter out any potential short results
    MIN_CHUNK_LENGTH_CHARS = 50 
    return [doc for doc in all_relevant_docs if len(doc.page_content) > MIN_CHUNK_LENGTH_CHARS]


# ==============================================================================
#                  3. FLASK WEB ROUTES (UPDATED)
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    # --- Determine search parameters from the request ---
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        per_page = request.form.get('per_page', 20, type=int)
        # Determine which button was clicked ('hybrid' or 'exact')
        search_type = request.form.get('search_type', 'hybrid') 
        page = 1
    else: # GET request for pagination
        query = request.args.get('query', '').strip()
        per_page = request.args.get('per_page', 20, type=int)
        search_type = request.args.get('search_type', 'hybrid')
        page = request.args.get('page', 1, type=int)

    # --- Perform the chosen search if a query exists ---
    if query:
        print(f"Handling search for '{query}' | Type: {search_type} | Page: {page}")
        
        if search_type == 'exact':
            all_results = perform_exact_search(query)
        else: # Default to hybrid search
            all_results = perform_hybrid_search(query)
        
        # --- Pagination Calculations ---
        total_results = len(all_results)
        total_pages = math.ceil(total_results / per_page)
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        results_for_page = all_results[start_index:end_index]
        
        pagination = {
            'page': page, 'per_page': per_page, 'total_pages': total_pages,
            'total_results': total_results, 'has_prev': page > 1,
            'has_next': page < total_pages, 'results': results_for_page
        }
        # Pass search_type back to the template to build correct pagination links
        return render_template('index.html', query=query, pagination=pagination, search_type=search_type)

    return render_template('index.html', query=None, pagination=None)


# ==============================================================================
#                  4. RUN THE FLASK APPLICATION (Unchanged)
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)