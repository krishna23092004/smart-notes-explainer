import streamlit as st
import fitz  # pymupdf
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import uuid
import shutil
from groq import Groq

# ==========================================
# 🔑 GROQ API SETUP
# ==========================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# 1. Initialize Memory + Unique Session ID
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Each user gets a unique session ID → their own private ChromaDB folder
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Build a unique persist path for this session
persist_path = f"./chroma_db_{st.session_state.session_id}"

# --- Sidebar for Uploading & Settings ---
with st.sidebar:
    st.title("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Notes", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    st.divider()
    st.markdown("### 📑 PDF Batch Processing")
    st.caption("Only applies to PDFs. Images scan the whole file.")
    col1, col2 = st.columns(2)
    with col1:
        start_page = int(st.number_input("Start", min_value=1, value=1))
    with col2:
        end_page = int(st.number_input("End", min_value=1, value=20))
    
    # --- Unified Analyze Logic ---
    if uploaded_file:
        if st.button("🔍 Deep Scan File"):
            with st.spinner("Extracting text..."):
                file_extension = uploaded_file.name.split('.')[-1].lower()
                extracted_text = ""
                
                try:
                    # LOGIC 1: IF FILE IS A PDF
                    if file_extension == 'pdf':
                        with open("temp_file.pdf", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        doc = fitz.open("temp_file.pdf")
                        for page_num in range(start_page - 1, min(end_page, len(doc))):
                            page = doc[page_num]
                            extracted_text += page.get_text()
                        doc.close()
                        
                        if os.path.exists("temp_file.pdf"):
                            os.remove("temp_file.pdf")
                            
                    # LOGIC 2: IF FILE IS AN IMAGE
                    elif file_extension in ['png', 'jpg', 'jpeg']:
                        image = Image.open(uploaded_file)
                        extracted_text = pytesseract.image_to_string(image)

                    # --- Process the Extracted Text ---
                    if not extracted_text.strip():
                        st.error("No text could be extracted. The image might be too blurry or blank.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents([Document(page_content=extracted_text)])
                        
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        
                        # ✅ FIX: Use session-specific persist_path so each user's data is isolated
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_path
                        )
                        st.success("✅ File successfully added to the Brain!")
                        
                        with st.expander("📝 See what the AI extracted"):
                            st.markdown(extracted_text[:1500] + "\n\n... (Text truncated)")
                        
                except Exception as e:
                    st.error(f"Error during scan: {e}")

    st.divider()
    st.header("🧠 AI Personality")
    study_mode = st.radio(
        "Choose how the AI responds:",
        ["Standard Assistant", "Explain Like I'm 5", "Quiz Master"]
    )

    st.divider()
    if st.button("🗑️ Clear Chat Memory"):
        st.session_state.messages = []
        # ✅ FIX: Also wipe this session's ChromaDB folder on clear
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.rerun()

# --- Main Chat UI ---
st.title("🤖 Smart Notes Explainer")

for message in st.session_state.messages:
    if message["role"] != "system": 
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "vectorstore" in st.session_state:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in docs])
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
                
                if study_mode == "Standard Assistant":
                    sys_prompt = "You are a helpful study assistant. Base your answers ONLY on the provided context."
                elif study_mode == "Explain Like I'm 5":
                    sys_prompt = "You are a teacher explaining concepts to a complete beginner. Use very simple, everyday analogies. Base your answers ONLY on the context."
                elif study_mode == "Quiz Master":
                    sys_prompt = "You are a strict examiner. Look at the context provided, and instead of answering the user's question, generate a difficult multiple-choice question based on the context to test their knowledge."

                try:
                    api_messages = [{'role': 'system', 'content': sys_prompt}]
                    for msg in st.session_state.messages[:-1]:
                        api_messages.append({'role': msg['role'], 'content': msg['content']})
                    
                    api_messages.append({'role': 'user', 'content': full_prompt})

                    chat_completion = client.chat.completions.create(
                        messages=api_messages,
                        model="llama-3.3-70b-versatile", 
                        temperature=0.3,
                    )
                    
                    answer = chat_completion.choices[0].message.content
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Exact Source Text (Anti-Hallucination)"):
                        st.caption("The AI used these specific chunks from your document to answer:")
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Snippet {i+1}:**")
                            st.info(doc.page_content)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Groq API Error: {e}")
            else:
                st.warning("Please upload a file and click 'Deep Scan File' first!")