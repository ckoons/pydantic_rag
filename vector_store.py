"""
Vector database module for the Pydantic RAG application.
Handles storage and retrieval of document embeddings using FAISS.
"""

import os
import numpy as np
import json
import pickle
import faiss
from typing import List, Dict, Tuple, Any, Optional, Union
import threading

# Vector store configuration
VECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_index")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")

# Create vector directory if it doesn't exist
os.makedirs(VECTOR_DIR, exist_ok=True)

# Lock for thread safety
index_lock = threading.Lock()

class VectorStore:
    """FAISS-based vector storage for document embeddings."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.index = None
        self.dimension = 1536  # Default OpenAI embedding dimension
        self.loaded = False
        self.metadata = {}  # Maps document IDs to metadata
        self.id_to_index = {}  # Maps document IDs to FAISS indices
        self.index_to_id = {}  # Maps FAISS indices to document IDs
        self.next_index = 0  # Next available index
        
        # Try to load existing index
        self._load()
    
    def _load(self) -> bool:
        """
        Load the vector index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        with index_lock:
            try:
                # Check if the index and metadata files exist
                if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
                    # Load the index
                    self.index = faiss.read_index(INDEX_PATH)
                    
                    # Load metadata
                    with open(METADATA_PATH, 'r') as f:
                        data = json.load(f)
                        self.metadata = data.get('metadata', {})
                        self.id_to_index = data.get('id_to_index', {})
                        self.index_to_id = {int(k): v for k, v in data.get('index_to_id', {}).items()}
                        self.dimension = data.get('dimension', 1536)
                        self.next_index = data.get('next_index', 0)
                    
                    self.loaded = True
                    return True
                else:
                    # Create a new index
                    self._create_new_index()
                    return False
            except Exception as e:
                print(f"Error loading vector index: {e}")
                # Create a new index if loading fails
                self._create_new_index()
                return False
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        with index_lock:
            # Create a new flat L2 index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0
            self.loaded = True
    
    def _save(self) -> bool:
        """
        Save the vector index and metadata to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        with index_lock:
            try:
                # Save the index
                faiss.write_index(self.index, INDEX_PATH)
                
                # Save metadata
                data = {
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'dimension': self.dimension,
                    'next_index': self.next_index
                }
                
                with open(METADATA_PATH, 'w') as f:
                    json.dump(data, f)
                
                return True
            except Exception as e:
                print(f"Error saving vector index: {e}")
                return False
    
    def add_document(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Add a document embedding to the vector store.
        
        Args:
            doc_id: Unique document identifier
            embedding: Vector embedding of the document
            metadata: Additional information about the document
            
        Returns:
            True if added successfully, False otherwise
        """
        with index_lock:
            try:
                # Make sure index is loaded
                if not self.loaded:
                    self._load()
                
                # Convert embedding to numpy array
                vector = np.array([embedding], dtype=np.float32)
                
                # If the document already exists, update it
                if doc_id in self.id_to_index:
                    old_index = self.id_to_index[doc_id]
                    # To update, we need to remove and re-add (FAISS doesn't support direct updates)
                    # This is a simplified approach - in production, you might want to batch updates
                    # For now, we'll just add a new vector and update the mappings
                    self.index.add(vector)
                    idx = self.next_index
                    self.next_index += 1
                    self.id_to_index[doc_id] = idx
                    self.index_to_id[idx] = doc_id
                    self.metadata[doc_id] = metadata
                else:
                    # Add new vector
                    self.index.add(vector)
                    idx = self.next_index
                    self.next_index += 1
                    self.id_to_index[doc_id] = idx
                    self.index_to_id[idx] = doc_id
                    self.metadata[doc_id] = metadata
                
                # Save the updated index
                self._save()
                return True
            except Exception as e:
                print(f"Error adding document to vector store: {e}")
                return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add multiple document embeddings to the vector store.
        
        Args:
            documents: List of documents, each with 'id', 'embedding', and 'metadata'
            
        Returns:
            True if added successfully, False otherwise
        """
        with index_lock:
            try:
                # Make sure index is loaded
                if not self.loaded:
                    self._load()
                
                # Prepare vectors and mappings
                vectors = []
                new_indices = {}
                
                for doc in documents:
                    doc_id = doc['id']
                    embedding = doc['embedding']
                    metadata = doc['metadata']
                    
                    # Add to vectors list
                    vectors.append(embedding)
                    
                    # Update mappings
                    idx = self.next_index
                    self.next_index += 1
                    new_indices[idx] = doc_id
                    self.id_to_index[doc_id] = idx
                    self.metadata[doc_id] = metadata
                
                # Convert to numpy array
                vectors_array = np.array(vectors, dtype=np.float32)
                
                # Add vectors to index
                self.index.add(vectors_array)
                
                # Update index to ID mapping
                self.index_to_id.update(new_indices)
                
                # Save the updated index
                self._save()
                return True
            except Exception as e:
                print(f"Error adding documents to vector store: {e}")
                return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        with index_lock:
            try:
                # Make sure index is loaded
                if not self.loaded:
                    self._load()
                
                # Check if index is empty
                if self.index.ntotal == 0:
                    return []
                
                # Convert query to numpy array
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # Search the index
                distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
                
                # Prepare results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1:  # FAISS returns -1 for empty slots
                        doc_id = self.index_to_id.get(int(idx))
                        if doc_id:
                            metadata = self.metadata.get(doc_id, {})
                            results.append({
                                'id': doc_id,
                                'score': float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
                                'metadata': metadata
                            })
                
                return results
            except Exception as e:
                print(f"Error searching vector store: {e}")
                return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        with index_lock:
            try:
                # Make sure index is loaded
                if not self.loaded:
                    self._load()
                
                # Check if document exists
                if doc_id not in self.id_to_index:
                    return False
                
                # For FAISS, we can't easily delete individual vectors
                # The simplest approach is to rebuild the index without the deleted document
                # This is inefficient for frequent deletions, but works for occasional ones
                
                # Remove from mappings
                idx = self.id_to_index[doc_id]
                del self.id_to_index[doc_id]
                if idx in self.index_to_id:
                    del self.index_to_id[idx]
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
                
                # Save the updated mappings
                # We're not actually removing from the FAISS index, just updating our mappings
                # The vector remains but becomes inaccessible - this is a tradeoff for simplicity
                self._save()
                return True
            except Exception as e:
                print(f"Error deleting document from vector store: {e}")
                return False
    
    def clear(self) -> bool:
        """
        Clear the vector store.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        with index_lock:
            try:
                # Create a new index
                self._create_new_index()
                
                # Save the empty index
                self._save()
                return True
            except Exception as e:
                print(f"Error clearing vector store: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        with index_lock:
            try:
                # Make sure index is loaded
                if not self.loaded:
                    self._load()
                
                return {
                    'num_documents': len(self.metadata),
                    'num_vectors': self.index.ntotal if self.index else 0,
                    'dimension': self.dimension,
                    'index_type': str(type(self.index)),
                    'index_path': INDEX_PATH,
                    'metadata_path': METADATA_PATH
                }
            except Exception as e:
                print(f"Error getting vector store stats: {e}")
                return {
                    'error': str(e)
                }


# Global instance
vector_store = VectorStore()