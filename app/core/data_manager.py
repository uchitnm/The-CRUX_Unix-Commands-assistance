import pandas as pd
from pathlib import Path
from app.config import settings

class DataManager:
    def __init__(self):
        self.df = None
        self.chunk_metadata = None
        self._load_data()
    
    def _load_data(self):
        """Load command data from CSV file."""
        print(f"Loading command data from {settings.DATA_PATH}...")
        self.df = pd.read_csv(settings.DATA_PATH)
        print(f"Loaded {len(self.df)} commands")
    
    def prepare_chunks(self):
        """Prepare chunks of the command data for more precise retrieval."""
        print(f"Preparing chunks with size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}...")
        chunks = []
        chunk_metadata = []
        
        for idx, row in self.df.iterrows():
            command = row['Command']
            description = str(row.get('DESCRIPTION', ''))
            examples = str(row.get('EXAMPLES', ''))
            options = str(row.get('OPTIONS', ''))
            
            # Combine all text for this command
            full_text = f"Command: {command}\nDescription: {description}\nExamples: {examples}\nOptions: {options}"
            
            # Create chunks with overlap
            for i in range(0, len(full_text), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
                chunk_text = full_text[i:i + settings.CHUNK_SIZE]
                if len(chunk_text) < 50:  # Skip very small chunks
                    continue
                    
                chunks.append(chunk_text)
                chunk_metadata.append({
                    'original_idx': idx,
                    'command': command,
                    'chunk_idx': len(chunks) - 1,
                    'text': chunk_text
                })
        
        self.chunk_metadata = pd.DataFrame(chunk_metadata)
        print(f"Created {len(chunks)} chunks")
        return self.chunk_metadata, chunks
    
    def save_chunk_metadata(self):
        """Save chunk metadata to file."""
        if self.chunk_metadata is not None:
            self.chunk_metadata.to_csv(settings.FAISS_METADATA_PATH, index=False)
            print(f"Saved chunk metadata to {settings.FAISS_METADATA_PATH}")
    
    def load_chunk_metadata(self):
        """Load chunk metadata from file."""
        if settings.FAISS_METADATA_PATH.exists():
            self.chunk_metadata = pd.read_csv(settings.FAISS_METADATA_PATH)
            print(f"Loaded chunk metadata with {len(self.chunk_metadata)} chunks")
            return self.chunk_metadata
        return None
    
    @property
    def commands(self):
        """Get the commands dataframe."""
        return self.df
    
    @property
    def metadata(self):
        """Get the chunk metadata."""
        return self.chunk_metadata 