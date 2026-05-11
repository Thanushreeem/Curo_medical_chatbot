Curo - AI Medical Chatbot
Curo is a medical chatbot I built using RAG (Retrieval Augmented Generation). It reads a medical PDF and answers health-related questions based on that document. I used Endee as the vector database to store and search embeddings.

Note: This chatbot is for informational purposes only. Always consult a real doctor for medical advice.

What it does : 
You can ask Curo questions like:
    What are the symptoms of covid?
    Remedies for fever
    What is diabetes?
    Causes of high blood pressure

It will give you a clear, simple answer based on the medical document.

Tech I used :
    Python - main language
    Flask - for the web app
    Groq API - LLaMA 3.3 70B model for generating answers
    HuggingFace - sentence-transformers for creating embeddings
    Endee - vector database to store and search embeddings
    LangChain - for building the RAG pipeline
    Docker - to run the Endee server locally

Project Structure : 
    tap_project/
        ├── endee/               # Endee vector database (forked repo)
            └── project/
        ├── Data/
        │   └── Medical_book.pdf
        ├── src/
        │   └── helper.py
        ├── static/
        │   ├── bot.png
        │   └── style.css
        ├── templates/
        │   └── chat.html
        ├── app.py
        ├── prompt.py
        ├── store_index.py
        ├── requirements.txt
        └── .env

How to run this project: 
    Step 1 - Clone the repo:
        git clone https://github.com/yourusername/tap_project.git
        cd tap_project/project

    Step 2 - Create virtual environment
        python -m venv venv

        # Windows
        venv\Scripts\activate.bat

        # Mac/Linux
        source venv/bin/activate    
    Step 3 - Install packages
        pip install -r requirements.txt
        pip install endee
    Step 4 - Add your API key
        Create a .env file in the project folder:
            GROQ_API_KEY=your_groq_api_key_here    #  Get free Groq API key from https://console.groq.com
    Step 5 - Start Endee vector database
        Make sure Docker is installed, then run:
            docker run -p 8080:8080 -v endee-data:/data endeeio/endee-server:latest
            Keep this terminal open.
    Step 6 - Upload PDF data to Endee
        Open a new terminal and run:
            python store_index.py
            Run this only once. It reads the PDF, creates embeddings and uploads them to Endee.
    Step 7 - Start the app
        python app.py
        Open browser and go to: http://127.0.0.1:5000

How it works :
The medical PDF is loaded and split into small chunks
Each chunk is converted to a vector using HuggingFace embeddings
All vectors are stored in Endee vector database
When user asks a question, it is also converted to a vector
Endee searches for the most similar chunks
Those chunks are sent to Groq LLM along with the question
LLM gives a clear answer based on the document


About Endee
Endee is a high performance open source vector database. I used it in this project to store and search medical text embeddings. It supports up to 1 billion vectors on a single node and is easy to run locally using Docker.
GitHub: https://github.com/endee-io/endee

Requirements
flask
python-dotenv
langchain
langchain-community
langchain-groq
langchain-huggingface
sentence-transformers
pypdf
endee

Important :Do not push your .env file to GitHub
Make sure Docker is running before starting the app
Run store_index.py only once unless you change the PDF
