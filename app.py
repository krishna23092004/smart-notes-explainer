import streamlit as st
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import os

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
            with st.spinner(f"Vision AI is reading Pages {start_page} to {end_page}..."):
                with open("temp_file.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    converter = DocumentConverter()
                    result = converter.convert("temp_file.pdf", page_range=(start_page, end_page))
                    markdown_text = result.document.export_to_markdown()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents([Document(page_content=markdown_text)])
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    
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

                # --- Fixed: removed extra closing parenthesis ---
                response = ollama.chat(model='llama3.2', messages=[
                    {'role': 'system', 'content': sys_prompt},
                    *st.session_state.messages,
                    {'role': 'user', 'content': full_prompt}
                ])
                
                answer = response['message']['content']
                st.markdown(answer)
                
                # Source Citations
                with st.expander("🔍 View Exact Source Text (Anti-Hallucination)"):
                    st.caption("The AI used these specific chunks from your document to answer:")
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Snippet {i+1}:**")
                        st.info(doc.page_content)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.warning("Please upload a file and click 'Deep Scan' first!")