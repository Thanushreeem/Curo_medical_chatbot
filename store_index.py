import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helper import download_hugging_face_embeddings
from src.helper import load_pdf_file, text_split

load_dotenv()

# LOAD & SPLIT PDF
print("Loading PDFs...")
docs = load_pdf_file("Data/")

print("Splitting text...")
chunks = text_split(docs)
print(f"Total chunks: {len(chunks)}")

# EMBEDDINGS
print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

# CONNECT TO PINECONE
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "chatbot"

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Created index: {INDEX_NAME}")
else:
    print(f"ℹ️ Index already exists: {INDEX_NAME}")

# UPLOAD TO PINECONE
print("Uploading chunks to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print(f"✅ All {len(chunks)} chunks uploaded to Pinecone successfully!")