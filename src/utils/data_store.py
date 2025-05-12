"""
Module to store and provide access to our data (embeddings, excerpts, index).
"""
import os
import pickle
import time
import hnswlib
import numpy as np

# Global variables to store the loaded data
global_excerpts = []
global_embeddings = []
global_index = None

def load_data():
    """Load the precomputed embeddings and index."""
    global global_excerpts, global_embeddings, global_index
    
    start_time = time.time()
    print("Loading precomputed embeddings and index...")
    
    # Paths to the data files
    excerpts_file = 'data/excerpts.pkl'
    embeddings_file = 'data/excerpt_embeddings.pkl'
    index_file = 'data/excerpt_index.bin'
    
    # Load excerpts
    if os.path.exists(excerpts_file):
        with open(excerpts_file, 'rb') as f:
            global_excerpts = pickle.load(f)
        print(f"Loaded {len(global_excerpts)} excerpts from {excerpts_file}")
    else:
        print(f"Warning: {excerpts_file} not found")
    
    # Load embeddings
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            global_embeddings = pickle.load(f)
        print(f"Loaded {len(global_embeddings)} embeddings from {embeddings_file}")
    else:
        print(f"Warning: {embeddings_file} not found")
    
    # Load HNSW index
    if os.path.exists(index_file):
        try:
            # Get the dimension from the first embedding
            if len(global_embeddings) > 0:
                dim = len(global_embeddings[0])
                
                # Initialize the index
                global_index = hnswlib.Index(space='cosine', dim=dim)
                
                # Load the index from file
                global_index.load_index(index_file)
                print(f"Loaded HNSW index from {index_file}")
                
                # Set search parameters
                global_index.set_ef(50)  # Higher values = more accurate but slower search
            else:
                print("Warning: Cannot load index without embeddings")
        except Exception as e:
            print(f"Error loading index: {e}")
    else:
        print(f"Warning: {index_file} not found")
    
    elapsed_time = time.time() - start_time
    print(f"Data loading completed in {elapsed_time:.2f} seconds")

# Provide a way for other modules to access the data
def get_excerpts():
    return global_excerpts

def get_embeddings():
    return global_embeddings

def get_index():
    return global_index