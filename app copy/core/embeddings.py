import faiss
from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingManager:
    def __init__(self):
        self.index = None
        self.embedding_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        print(f"Initializing embedding model {settings.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    def create_index(self, texts):
        """Create FAISS index from text embeddings."""
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Created embeddings with dimension {dimension}")
        return self.index
    
    def load_index(self):
        """Load FAISS index from file."""
        if settings.FAISS_INDEX_PATH.exists():
            print(f"Loading existing FAISS index from {settings.FAISS_INDEX_PATH}...")
            self.index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
        return False
    
    def save_index(self):
        """Save FAISS index to file."""
        if self.index is not None:
            faiss.write_index(self.index, str(settings.FAISS_INDEX_PATH))
            print(f"Saved FAISS index to {settings.FAISS_INDEX_PATH}")
    
    def search(self, query, top_n=None):
        """Search the index for similar vectors."""
        if self.index is None:
            raise ValueError("No index loaded. Call load_index() or create_index() first.")
        
        if top_n is None:
            top_n = settings.TOP_N_RESULTS
            
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_n)
        return distances[0], indices[0]
    
    @property
    def model(self):
        """Get the embedding model."""
        return self.embedding_model 