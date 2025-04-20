
import os
from langchain.tools.tavily_search import TavilySearchResults
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def run_research_agent(query):
    print(f"🔍 Researching: {query}")
    
    # Search with Tavily
    tavily = TavilySearchResults(api_key=TAVILY_API_KEY)
    search_results = tavily.run(query)
    
    # Extract URLs
    urls = [item['url'] for item in search_results['results'][:3]]  # Top 3 results
    
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            page_docs = loader.load()
            documents.extend(page_docs)
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    
    # Split and summarize documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Embed and store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("deep-research-agent/data/processed/vector_store")

    print(f"✅ Stored {len(docs)} chunks into vector store.")
