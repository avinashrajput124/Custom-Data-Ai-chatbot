from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_community.document_loaders import PDFMinerLoader

def main():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing file: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        db = Chroma.from_documents(texts, embeddings, persist_directory="db")
        print(f"Chroma vector store created: {db}")
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")

if __name__ == "__main__":
    main()


