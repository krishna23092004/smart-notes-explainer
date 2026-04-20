import streamlit as st
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# Ollama ki jagah free HuggingFace embeddings (Cloud par smoothly chalega)
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
# Groq API import
from groq import Groq

# ==========================================
# 🔑 GROQ API SETUP (Yahan apni API key dalein)
# ==========================================
# Note: Jab aap ise cloud par deploy karein, toh st.secrets ka use karna best practice hai
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]   
client = Groq(api_key=GROQ_API_KEY)

# 1. Initialize Memory in the storage locker
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Uploading & Settings ---
with st.sidebar:
    st.title("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Notes", type=['pdf', 'png', 'jpg'])
    
    # Batch Processing Selectors
    st.divider()
    st.markdown("### 📑 Batch Processing")
    st.caption("For large PDFs, scan in chunks to prevent crashes (e.g., 1-20, then 21-40).")
    
    col1, col2 = st.columns(2)
    with col1:
        start_page = int(st.number_input("Start", min_value=1, value=1))
    with col2:
        end_page = int(st.number_input("End", min_value=1, value=20))
    
    # --- Unified Analyze Logic ---
    if uploaded_file:
        if st.button(f"🔍 Deep Scan (Pages {start_page}-{end_page})"):
            with st.spinner(f"Reading Pages {start_page} to {end_page}..."):
                with open("temp_file.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # PDF Reading via PyMuPDF
                    doc = fitz.open("temp_file.pdf")
                    markdown_text = ""
                    for page_num in range(start_page - 1, min(end_page, len(doc))):
                        page = doc[page_num]
                        markdown_text += page.get_text()
                    doc.close()

                    if not markdown_text.strip():
                        st.error("No text could be extracted from the selected pages.")
                    else:
                        # Splitting Text
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents([Document(page_content=markdown_text)])
                        
                        # Updated: Using HuggingFace Embeddings instead of Ollama
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        
                        # Creating Vector Database
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory="./chroma_db"
                        )
                        st.success(f"✅ Pages {start_page} to {end_page} successfully added to the Brain!")
                        
                        with st.expander("📝 See what the AI extracted"):
                            st.markdown(markdown_text[:1500] + "\n\n... (Text truncated)")
                        
                except Exception as e:
                    st.error(f"Error during scan: {e}")
                
                finally:
                    if os.path.exists("temp_file.pdf"):
                        os.remove("temp_file.pdf")

    st.divider()
    
    # --- AI Personality Selection ---
    st.header("🧠 AI Personality")
    study_mode = st.radio(
        "Choose how the AI responds:",
        ["Standard Assistant", "Explain Like I'm 5", "Quiz Master"]
    )

    st.divider()
    if st.button("🗑️ Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat UI ---
st.title("🤖 Smart Notes Explainer")

# Display previous messages from memory
for message in st.session_state.messages:
    if message["role"] != "system": # Hide system messages from UI
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- The Chat Input ---
if prompt := st.chat_input("Ask me anything about your notes..."):
    # Add user message to UI and Memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Logic to generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "vectorstore" in st.session_state:
                # Retrieve top snippets
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
                
                # --- Dynamic System Prompts ---
                if study_mode == "Standard Assistant":
                    sys_prompt = "You are a helpful study assistant. Base your answers ONLY on the provided context."
                elif study_mode == "Explain Like I'm 5":
                    sys_prompt = "You are a teacher explaining concepts to a complete beginner. Use very simple, everyday analogies. Base your answers ONLY on the context."
                elif study_mode == "Quiz Master":
                    sys_prompt = "You are a strict examiner. Look at the context provided, and instead of answering the user's question, generate a difficult multiple-choice question based on the context to test their knowledge."

                # --- Groq API Call (Replaced Ollama) ---
                try:
                    # Construct message history for Groq
                    api_messages = [{'role': 'system', 'content': sys_prompt}]
                    # Ignore the last user message in history because we append full_prompt below
                    for msg in st.session_state.messages[:-1]:
                        api_messages.append({'role': msg['role'], 'content': msg['content']})
                    
                    api_messages.append({'role': 'user', 'content': full_prompt})

                    # Call Groq (Using Llama 3.1 70B for high quality responses)
                    chat_completion = client.chat.completions.create(
                        messages=api_messages,
                        model="llama-3.1-70b-versatile",
                        temperature=0.3,
                    )
                    
                    answer = chat_completion.choices[0].message.content
                    st.markdown(answer)
                    
                    # Source Citations
                    with st.expander("🔍 View Exact Source Text (Anti-Hallucination)"):
                        st.caption("The AI used these specific chunks from your document to answer:")
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Snippet {i+1}:**")
                            st.info(doc.page_content)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Groq API Error: {e}\nCheck your API key or internet connection.")
            else:
                st.warning("Please upload a file and click 'Deep Scan' first!")