import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Flask
from flask import Flask, render_template, request

# Env
from dotenv import load_dotenv

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Your files
from helper import download_hugging_face_embeddings
from prompt import system_prompt

# ===== LOAD ENV =====
load_dotenv()

app = Flask(__name__)

# ===== EMBEDDINGS =====
embeddings = download_hugging_face_embeddings()

# ===== PINECONE INDEX =====
index_name = "chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ===== GROQ LLM =====
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.4
)

# ===== PROMPT =====
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ===== RAG CHAIN =====
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    try:
        response = rag_chain.invoke({"input": msg})
        return response["answer"]
    except Exception as e:
        print("ERROR:", e)
        return "Error occurred. Check server logs."

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)