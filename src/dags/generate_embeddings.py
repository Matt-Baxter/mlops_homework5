"""
Script to generate embeddings for Wikipedia excerpts and create a searchable index.
This is a batch job that only needs to be run once to prepare the data.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import time

# Output file paths - relative to project root
EMBEDDINGS_FILE = 'data/excerpt_embeddings.pkl'
EXCERPTS_FILE = 'data/excerpts.pkl'
CORPUS_FILE = 'data/corpus.pkl'

def main():
    print("Starting embedding generation process...")
    start_time = time.time()
    
    # Step 1: Load the dataset
    print("Loading dataset...")
    csv_path = 'data/6000_all_categories_questions_with_excerpts.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return
        
    # Use pandas to read the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Step 2: Extract the excerpts
    print("Extracting excerpts...")
    
    # The dataset has a column 'wikipedia_excerpt' which contains the Wikipedia content
    excerpts = df['wikipedia_excerpt'].tolist()
    
    # Keep the first 1000 characters if excerpts are too long
    excerpts = [excerpt[:1000] for excerpt in excerpts]
    print(f"Extracted {len(excerpts)} excerpts")
    
    # Step 3: Load the sentence transformer model
    print("Loading the sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Model loaded successfully")
    
    # Step 4: Generate embeddings
    print("Generating embeddings (this may take a while)...")
    embeddings = model.encode(excerpts, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Step 5: Save embeddings and excerpts
    print("Saving embeddings and excerpts...")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    
    # Save embeddings
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save excerpts
    with open(EXCERPTS_FILE, 'wb') as f:
        pickle.dump(excerpts, f)
    
    # Save both together as a corpus dictionary
    corpus = {
        'excerpts': excerpts,
        'embeddings': embeddings
    }
    with open(CORPUS_FILE, 'wb') as f:
        pickle.dump(corpus, f)
    
    # Optional: Create HNSW index
    # run 'pip install hnswlib' in venv to install
    try:
        import hnswlib
        print("Creating HNSW index...")
        
        # Initialize HNSW index
        dim = embeddings.shape[1]  # Embedding dimension
        num_elements = len(embeddings)
        
        # Create and configure index
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Add embeddings to index
        index.add_items(embeddings, list(range(len(embeddings))))
        
        # Save index
        index.save_index('data/excerpt_index.bin')
        print("HNSW index created and saved successfully")
    except ImportError:
        print("hnswlib not installed. Skipping HNSW index creation.")
        print("To install: pip install hnswlib")
    
    elapsed_time = time.time() - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds")
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Excerpts saved to {EXCERPTS_FILE}")
    print(f"Corpus saved to {CORPUS_FILE}")

if __name__ == "__main__":
    main()