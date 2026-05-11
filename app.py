import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from endee import Endee
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings
from prompt import PROMPT

load_dotenv()

app = Flask(__name__)

#  EMBEDDINGS 
embeddings_model = download_hugging_face_embeddings()

#  CONNECT TO ENDEE 
client = Endee()

index = client.get_index(name="medicalchatbot")

#  LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=512
)

def retrieve_context(query, top_k=6):
    """Search Endee for relevant chunks"""
    query_vector = embeddings_model.embed_query(query)
    results = index.query(vector=query_vector, top_k=top_k)
    context = "\n\n".join([r["meta"]["text"] for r in results if "meta" in r])
    return context

def get_answer(query):
    """Get answer from LLM using retrieved context"""
    context = retrieve_context(query)
    prompt_text = PROMPT.format(context=context, question=query)
    response = llm.invoke(prompt_text)
    return response.content

# ROUTES
@app.route("/")
def index_page():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    try:
        result = get_answer(msg)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please try again."

#  RUN 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)