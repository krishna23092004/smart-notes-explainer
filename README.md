Smart Notes Explainer 🤖
Ever stared at a 50-page PDF the night before an exam and wished you could just talk to it? That's exactly why I built this.
Smart Notes Explainer lets you upload your notes (PDF or image) and ask questions about them in plain English. It reads your document, understands it, and answers based only on what's actually written — no random AI hallucinations.
🔗 Try it live  |  💻 Source code

What it can do

Upload a PDF or a photo of your handwritten notes
Ask anything — "what is the main topic of chapter 3?" or "explain this concept simply"
See exactly which part of your document the AI used to answer (so you know it's not making things up)
Switch between 3 modes depending on how you want to study:

Standard — just answers your question
ELI5 — explains like you're a complete beginner
Quiz Master — turns your notes into MCQs to test yourself


Works for multiple users at once — your uploaded file stays private to your session


How it works (roughly)
When you upload a file, the app extracts all the text (using OCR for images, PyMuPDF for PDFs), breaks it into small chunks, converts those chunks into vector embeddings, and stores them in ChromaDB. When you ask a question, it finds the most relevant chunks and passes them to the LLM along with your question. The LLM then answers strictly based on that context.
This pattern is called RAG (Retrieval-Augmented Generation) — it's what keeps the AI grounded to your actual document instead of guessing.

Tech used

Streamlit — for the UI
Groq API (Llama 3.3 70B) — the language model
ChromaDB — vector database to store and search embeddings
HuggingFace all-MiniLM-L6-v2 — for generating embeddings
Tesseract + PyMuPDF — text extraction from images and PDFs
LangChain — to wire everything together


Running it locally
bashgit clone https://github.com/krishna23092004/smart-notes-explainer.git
cd smart-notes-explainer
pip install -r requirements.txt
You'll also need Tesseract installed on your system:
bash# Ubuntu/Debian
sudo apt-get install tesseract-ocr libgl1 libglib2.0-0

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
Then create a .streamlit/secrets.toml file:
tomlGROQ_API_KEY = "your_key_here"
Get a free Groq key at console.groq.com/keys — takes 2 minutes.
Then run:
bashstreamlit run app.py

One thing I had to figure out
When I shared the app link with a friend to test it, I noticed their uploaded admit card was showing up in my session too. Turns out ChromaDB was writing everything to the same folder on the server, so all users were sharing one database.
Fixed it by giving each session its own folder using a UUID:
pythonif "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

persist_path = f"./chroma_db_{st.session_state.session_id}"
Simple fix, but easy to miss when you're building for single-user use first.

Project structure
smart-notes-explainer/
├── app.py
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── secrets.toml
└── README.md

Built by
Krishna — 3rd year B.Tech IT student at SGSITS Indore
GitHub · HuggingFace