from langchain_ollama import OllamaEmbeddings

#return for use of the local ollama with the llama3.1 model
def get_embedding_function():
    
    return OllamaEmbeddings(model="nomic-embed-text")