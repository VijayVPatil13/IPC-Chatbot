
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP 1: Configuration

PDF_PATH = "data/indian_penal_code.pdf"
VECTORSTORE_PATH = "vectorstore/ipc_faiss"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        "IPC PDF not found. Place it inside the data folder."
    )

# STEP 2: Load PDF

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Total PDF pages loaded:", len(documents))

# STEP 3: Preserve document structure during preprocessing

processed_documents = []

for doc in documents:
    text = doc.page_content

    # Preserve line breaks; normalize only horizontal whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # Remove unnecessary spaces around lines.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines without removing paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if text:
        doc.page_content = text
        processed_documents.append(doc)

documents = processed_documents

print("Pages after preprocessing:", len(documents))

# STEP 4: Split documents while preserving page metadata

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=300,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
        ""
    ],
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

# Add chunk metadata
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index
    chunk.metadata["page_number"] = (
        chunk.metadata.get("page", -1) + 1
        if isinstance(chunk.metadata.get("page"), int)
        else "Unknown"
    )
    chunk.metadata["source"] = PDF_PATH

print("Total chunks generated:", len(chunks))

# STEP 5: Inspect generated chunks

for chunk in chunks[:5]:
    print("\n========== CHUNK ==========")
    print("Chunk ID:", chunk.metadata["chunk_id"])
    print("Page:", chunk.metadata["page_number"])
    print("Start index:", chunk.metadata.get("start_index"))
    print(chunk.page_content)
    print("=" * 60)

# STEP 6: Generate embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 7: Create FAISS vector database

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# STEP 8: Save vector database

os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)

vectorstore.save_local(VECTORSTORE_PATH)

print("\nIPC document successfully processed!")
print("Total pages:", len(documents))
print("Total chunks:", len(chunks))
print("FAISS vector database created:", VECTORSTORE_PATH)
