# Indian Penal Code Chatbot

A Streamlit chatbot that answers questions using the Indian Penal Code (IPC) PDF included with this project. It uses retrieval-augmented generation (RAG): relevant passages are retrieved from a local FAISS index and sent to a Groq-hosted language model to generate a response. Retrieved PDF page numbers are shown with the answer.

> **Disclaimer:** This is an educational project, not legal advice. The supplied document represents the historical IPC and does not establish the current state of Indian criminal law. Verify legal questions with authoritative, up-to-date sources and a qualified professional.

## Requirements

- Python 3.10 or later
- A Groq API key
- Internet access for the Groq API and, on first ingestion, downloading the Hugging Face embedding model

## Setup

Run these commands from the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` private; it is excluded from Git.

## Build the local document index

The generated FAISS index is not included in Git. Create it from the supplied PDF before launching the app:

```powershell
python ingest.py
```

This reads `data/indian_penal_code.pdf`, splits and embeds its text using `sentence-transformers/all-MiniLM-L6-v2`, and writes the index to `vectorstore/ipc_faiss/`. Re-run ingestion if you replace or update the PDF.

## Run the chatbot

```powershell
streamlit run app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`) where you can open the chatbot.

## Project structure

```text
.
├── app.py                       # Streamlit chat interface and RAG pipeline
├── ingest.py                    # PDF preprocessing and FAISS index generation
├── requirements.txt             # Python dependencies
├── data/
│   └── indian_penal_code.pdf    # Source document
└── vectorstore/
    └── ipc_faiss/               # Generated locally by ingest.py; not committed
```

## Notes

- The app uses the `openai/gpt-oss-20b` model through Groq and retrieves up to six similar document chunks per question.
- Answers are intended to be grounded in the supplied document. Retrieval can be incomplete, so check the cited PDF pages.
- The app loads the local FAISS index with pickle deserialization enabled. Use only an index generated from a trusted source; do not load an untrusted index.
