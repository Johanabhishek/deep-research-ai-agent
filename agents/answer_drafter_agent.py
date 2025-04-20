
import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def run_drafter_agent(query):
    print(f"🧠 Generating answer for: {query}")

    # Load the vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local("deep-research-agent/data/processed/vector_store", embeddings)

    # Setup RetrievalQA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Get answer
    result = qa_chain(query)
    print("✅ Answer:")
    print(result["result"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")
