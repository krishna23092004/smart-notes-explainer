# 📚 Smart Notes Explainer (Local RAG)

An advanced, fully local Retrieval-Augmented Generation (RAG) web application designed to process complex academic documents, handwritten notes, and heavy mathematical textbooks (like DSP manuals) without relying on paid cloud APIs.

## 🌟 Key Features
* **100% Local AI:** Powered by `Ollama` running `Llama 3.2` and `nomic-embed-text` ensuring complete data privacy and $0 API costs.
* **Multimodal Vision Parsing:** Uses IBM's `Docling` to read text, math, and tables from heavy PDFs.
* **Batch Processing Engine:** Engineered a UI to process 400+ page textbooks in safe chunks to prevent memory overload.
* **Persistent Memory:** Utilizes `ChromaDB` for local vector storage, preventing the need to re-scan massive files.
* **Anti-Hallucination Measures:** Direct source-citation UI that exposes the raw vector data to the user.
* **Dynamic AI Personas:** Switch between Standard Assistant, ELI5 (Explain Like I'm 5), and Quiz Master modes.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Document Processing:** IBM Docling
