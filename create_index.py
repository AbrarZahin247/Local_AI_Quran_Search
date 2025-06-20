# ==============================================================================
#                      One-Time Search Index Creator
#
#  Run this script only once to build the necessary FAISS vector search index.
# ==============================================================================

import os
import ast
from itertools import groupby
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- Configuration (Should match app.py) ---
DATA_FILE_PATH = "quran_text.txt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = "faiss_index_bangla"
CHUNK_SIZE_IN_VERSES = 3

def create_and_save_index():
    """
    Reads the source text, processes it into chunks, creates vector embeddings,
    and saves the final FAISS index to disk.
    """
    print("===================================================")
    print("Starting the one-time index creation process...")
    print("This may take several minutes depending on your file size and CPU.")
    print("===================================================")

    # --- Step 1: Load and Process the Source Text File with Smart Chunking ---
    print(f"-> Loading data from '{DATA_FILE_PATH}'...")
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        
        initial_documents = []
        for line in raw_lines:
            try:
                line_tuple = ast.literal_eval(line.strip().rstrip(','))
                doc_id, surah_no, ayah_no, text = line_tuple
                initial_documents.append(Document(page_content=text, metadata={'surah': surah_no, 'ayah': ayah_no, 'id': doc_id}))
            except (ValueError, SyntaxError):
                continue
        
        if not initial_documents:
            print("!!! ERROR: No documents were loaded. Please check your quran_text.txt file.")
            return

        print(f"--> Successfully loaded {len(initial_documents)} raw verses.")
    except FileNotFoundError:
        print(f"!!! FATAL ERROR: Data file not found at '{DATA_FILE_PATH}'.")
        return

    # --- Step 2: Apply Smart Chunking Logic to Group Verses ---
    print(f"-> Applying smart chunking to group verses into chunks of {CHUNK_SIZE_IN_VERSES}...")
    documents = []
    groups = groupby(initial_documents, key=lambda doc: doc.metadata['surah'])
    for surah_no, surah_docs_iterator in groups:
        surah_docs = list(surah_docs_iterator)
        for i in range(0, len(surah_docs), CHUNK_SIZE_IN_VERSES):
            chunk_of_docs = surah_docs[i:i + CHUNK_SIZE_IN_VERSES]
            combined_text = " ".join([d.page_content for d in chunk_of_docs])
            start_ayah, end_ayah = chunk_of_docs[0].metadata['ayah'], chunk_of_docs[-1].metadata['ayah']
            new_meta = {'surah': surah_no, 'ayah_range': f"{start_ayah}-{end_ayah}"}
            new_doc = Document(page_content=combined_text, metadata=new_meta)
            documents.append(new_doc)
    print(f"--> Grouped into {len(documents)} context-rich documents.")

    # --- Step 3: Create Vector Embeddings and FAISS Index ---
    print(f"-> Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("-> Creating FAISS vector store from documents. This is the slowest step...")
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # --- Step 4: Save the Index to Disk ---
    print(f"-> Saving the index to disk at '{FAISS_INDEX_PATH}'...")
    vector_store.save_local(FAISS_INDEX_PATH)
    
    print("\n===================================================")
    print("      SUCCESS! Index has been created.             ")
    print(f"      You can now run the main application using 'python app.py'")
    print("===================================================")


if __name__ == "__main__":
    # If a faiss_index_bangla folder already exists, delete it to ensure a fresh build.
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Warning: An existing index was found at '{FAISS_INDEX_PATH}'.")
        user_input = input("Do you want to delete it and rebuild? (y/n): ").lower()
        if user_input == 'y':
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH)
            print("Old index deleted.")
            create_and_save_index()
        else:
            print("Index creation cancelled by user.")
    else:
        create_and_save_index()