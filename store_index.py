import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from endee import Endee, Precision
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

load_dotenv()

#  LOAD & SPLIT PDF 
print("Loading PDFs...")
docs = load_pdf_file("Data/")

print("Splitting text...")
chunks = text_split(docs)
print(f"Total chunks: {len(chunks)}")

#  EMBEDDINGS
print("Loading embeddings model...")
embeddings_model = download_hugging_face_embeddings()

# CONNECT TO ENDEE 
print("Connecting to Endee vector database")
client = Endee()

INDEX_NAME = "medicalchatbot"  
DIMENSION = 768

# Create index if not exists
try:
    response = client.list_indexes()
    indexes = response.get("indexes", [])
    existing = [idx.get("name", "") for idx in indexes] if indexes else []
    print(f"Existing indexes: {existing}")
except Exception as e:
    print(f"Could not list indexes: {e}")
    existing = []

if INDEX_NAME not in existing:
    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.INT8
    )
    print(f"✅ Created index: {INDEX_NAME}")
else:
    print(f"ℹ️ Index already exists: {INDEX_NAME}")

index = client.get_index(name=INDEX_NAME)

#  EMBED & UPSERT
print("Embedding and uploading chunks to Endee...")
vectors = []
for i, chunk in enumerate(chunks):
    vector = embeddings_model.embed_query(chunk.page_content)
    vectors.append({
        "id": f"chunk{i}",
        "vector": vector,
        "meta": {
            "text": chunk.page_content,
            "source": str(chunk.metadata.get("source", ""))
        }
    })

    if len(vectors) == 100:
        index.upsert(vectors)
        print(f"Uploaded {i+1} chunks...")
        vectors = []

if vectors:
    index.upsert(vectors)

print(f"✅ All {len(chunks)} chunks uploaded to Endee successfully!")