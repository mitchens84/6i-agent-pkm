# AGENT-PKM Embedding and Vector Search Configuration

# Default embedding settings
EMBEDDING_MODELS = {
    "default": {
        "name": "all-MiniLM-L6-v2",  # Current default
        "dimensions": 768,
        "provider": "sentence_transformers",
    },
    "improved": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "provider": "sentence_transformers",
        "description": "Better balanced semantic search model with improved retrieval accuracy",
    },
    "openai": {
        "name": "text-embedding-ada-002",
        "dimensions": 1536,
        "provider": "openai",
        "description": "OpenAI's embedding model with strong semantic understanding",
    },
    "e5": {
        "name": "intfloat/e5-large-v2",
        "dimensions": 1024,
        "provider": "sentence_transformers",
        "description": "Optimized for retrieval tasks and better handling of longer contexts",
    },
}

# Current active model - can be changed without code modification
ACTIVE_EMBEDDING_MODEL = "default"

# Pinecone settings
PINECONE_SETTINGS = {
    "namespace": "pkm_knowledge_base",
    "top_k": 5,  # Number of results to return in similarity search
    "include_metadata": True,
    "include_values": False,
}

# Chunking settings
CHUNKING_SETTINGS = {"chunk_size": 500, "chunk_overlap": 50}

# Reindexing settings
REINDEXING_SETTINGS = {
    "batch_size": 100,
    "sleep_interval": 0.5,  # Time to sleep between batches to avoid rate limiting
}
