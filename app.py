# ==============================================================================
#                      Bangla Semantic Search Engine - app.py
#           (Version with Dynamic Pagination and No AI Explanation)
# ==============================================================================

import os
import sys
import ast
import math
from itertools import groupby
from flask import Flask, render_template, request

# --- Import Core Libraries ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi

# ==============================================================================
#                      1. ONE-TIME INITIALIZATION
# ==============================================================================

print("===================================================")
print("Starting Flask server and loading pre-built models...")
print("===================================================")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
DATA_FILE_PATH = "quran_text.txt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = "faiss_index_bangla"
CHUNK_SIZE_IN_VERSES = 3
# ** RESULTS_PER_PAGE constant is now removed, as it's dynamic **

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
print("===================================================")

# ==============================================================================
#                  2. HYBRID SEARCH FUNCTION (Unchanged)
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
        fused_scores = {}
        doc_map = {doc.page_content: doc for results in results_lists for doc, _ in results}
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

# ==============================================================================
#          3. FLASK WEB ROUTES (UPDATED WITH DYNAMIC PER_PAGE LOGIC)
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    # --- Get parameters from the request ---
    if request.method == 'POST':
        # This is a new search submitted from the form
        query = request.form.get('query', '').strip()
        per_page = request.form.get('per_page', 20, type=int) # Get from form
        page = 1 # A new search always starts at page 1
    else:
        # This is a GET request, for navigating pages of a previous search
        query = request.args.get('query', '').strip()
        per_page = request.args.get('per_page', 20, type=int) # Get from URL
        page = request.args.get('page', 1, type=int)

    # --- Perform search if a query exists ---
    if query:
        print(f"Handling search for '{query}', page {page}, {per_page} per page")
        all_results = perform_hybrid_search(query)
        
        # --- Pagination Calculations using the dynamic per_page value ---
        total_results = len(all_results)
        total_pages = math.ceil(total_results / per_page)
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        
        results_for_page = all_results[start_index:end_index]
        
        # Create a pagination object to pass to the template
        pagination = {
            'page': page,
            'per_page': per_page, # Pass the current per_page setting
            'total_pages': total_pages,
            'total_results': total_results,
            'has_prev': page > 1,
            'has_next': page < total_pages,
            'results': results_for_page
        }
        return render_template('index.html', query=query, pagination=pagination)

    # Render the home page without any search results
    return render_template('index.html', query=None, pagination=None)


# ==============================================================================
#                  4. RUN THE FLASK APPLICATION (Unchanged)
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)