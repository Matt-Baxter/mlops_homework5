"""
Retriever module for finding similar responses to a query.
"""
from sentence_transformers import SentenceTransformer
from src.utils.data_store import get_excerpts, get_embeddings, get_index
import numpy as np

# Load the model once when the module is imported
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_similar_responses(question: str, top_k: int = 3) -> list:
    """
    Find the most similar responses to the given question.
    
    Args:
        question: The question to find similar responses for
        top_k: Number of top responses to return
        
    Returns:
        List of the top_k most similar excerpts
    """
    # Step 1: Convert question to embedding
    question_embedding = model.encode(question)
    
    # Get data from data_store module
    excerpts = get_excerpts()
    
    # Ensure we have data to work with
    if not excerpts:
        return ["No data available"]
    
    # Step 2: Compute similarity
    # There are two approaches - we'll use both, one as a backup
    
    # Approach 1: Use the HNSW index for fast retrieval (preferred for large datasets)
    index = get_index()
    if index:
        # Get nearest neighbors (returns indexes and distances)
        labels, distances = index.knn_query(question_embedding, k=top_k)
        
        # Get the corresponding excerpts
        similar_excerpts = [excerpts[idx] for idx in labels[0]]
        
        # Print some debug info
        print(f"Question: {question}")
        print(f"Top {top_k} similar excerpts found with distances: {distances[0]}")
        
        return similar_excerpts
    
    # Approach 2: Direct cosine similarity (fallback if index is not available)
    else:
        # Get embeddings
        embeddings = get_embeddings()
        
        if not embeddings:
            return ["No embeddings available"]
        
        # Convert to numpy for fast calculations
        embeddings_np = np.array(embeddings)
        question_embedding_np = np.array(question_embedding)
        
        # Normalize for cosine similarity
        embeddings_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        question_embedding_norm = question_embedding_np / np.linalg.norm(question_embedding_np)
        
        # Compute cosine similarity
        similarities = np.dot(embeddings_norm, question_embedding_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get the corresponding excerpts
        similar_excerpts = [excerpts[idx] for idx in top_indices]
        
        # Print some debug info
        print(f"Question: {question}")
        print(f"Top {top_k} similar excerpts found with similarities: {similarities[top_indices]}")
        
        return similar_excerpts