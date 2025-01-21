import logging
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
def load_documents():
    logging.info("Loading documents from directory...")
    loader = PyPDFDirectoryLoader("C:\\Users\\venka\\Desktop\\doctorproj\\contents")
    docs = loader.load()
    logging.info(f"Loaded {len(docs)} documents.")
    return docs
vectorstore_file = "vectorstoreexample.index"
if os.path.exists(vectorstore_file):
    logging.info("Loading existing vector store...")
    vectorstore = FAISS.load_local(vectorstore_file)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
else:
    logging.info("Documents not found. Processing new documents...")
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    logging.info("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logging.info(f"Saving vector store to {vectorstore_file}...")
    vectorstore.save_local(vectorstore_file)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

 # type: ignore