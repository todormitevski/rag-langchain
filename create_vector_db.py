from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import os
import shutil
import onnxruntime

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
CHROMA_PATH = "chroma/"

def main():
    loaded_docs = load_docs()
    created_chunks = split_text(loaded_docs)
    create_chroma_db(created_chunks)


def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="pyke_lore.md")
    docs = loader.load()
    return docs


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Successfully split {len(docs)} documents into {len(chunks)} chunks.")

    doc_chunk = chunks[2]
    print(doc_chunk.page_content)
    print(doc_chunk.metadata)

    return chunks


def create_chroma_db(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        # OpenAIEmbeddings(),
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        # this will automatically persist the db
        persist_directory=CHROMA_PATH
    )

    print(f"Persisted {len(chunks)} chunks to {CHROMA_PATH}.")

    
if __name__=="__main__":
    main()
