import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


def main():
    print("Loading documents...")
    loader=TextLoader("mediumblog.txt" , encoding="utf8")
    documents=loader.load()

    print("Splitting documents into chunks...")
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts=text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(texts)}")

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
    print("Ingesting")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))  
    print("finish")

if __name__ == "__main__":
    main()
