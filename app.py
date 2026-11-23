
import os
import re
import html
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# STEP 1: Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Groq API key not found in .env file.")
    st.stop()

# STEP 2: Configure Streamlit
st.set_page_config(page_title="IPC Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Indian Penal Code Chatbot")
st.write("Ask questions about the Indian Penal Code, its sections, offences and punishments.")

# STEP 3: Load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STEP 4: Load FAISS vector database
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        "vectorstore/ipc_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# STEP 5: Configure retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# STEP 6: Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=4096,
    reasoning_effort="low",
    reasoning_format="hidden",
    api_key=api_key
)

# STEP 7: Define RAG prompt

prompt = ChatPromptTemplate.from_template("""
You are an IPC legal-document research assistant.

Your task is to answer user questions using ONLY the supplied
Indian Penal Code (IPC) document context.

STRICT RULES:

1. LEGAL ACCURACY
   - Never invent IPC sections, legal provisions, punishments,
     definitions, illustrations or legal interpretations.
   - Use the retrieved document context as the sole source
     of legal facts.
   - Do not introduce external legal knowledge.
   - Do not modify or misrepresent the meaning of statutory text.

2. QUESTION RELEVANCE AND SCOPE
   - Identify exactly what the user is asking before generating
     an answer.
   - Answer only the specific question asked.
   - Include only information that directly helps answer it.
   - Do not introduce unrelated offences, IPC sections,
     legal concepts or examples.
   - Do not automatically discuss related offences.
   - Do not automatically provide comparisons or differences.
   - Include comparisons only when explicitly requested.
   - Do not provide definitions, ingredients, illustrations,
     exceptions or punishments unless they are requested
     or necessary to answer the question.
   - Do not expand a narrow question into a general discussion
     of the surrounding chapter or other provisions.

3. SECTION VERIFICATION
   - When asked about a specific section, verify that the exact
     section number appears in the retrieved context.
   - Verify that the retrieved text actually describes
     the requested section.
   - Do not mistake references to another section for its
     definition or punishment.
   - Keep the definition, ingredients, punishment,
     illustrations and exceptions associated with their
     respective provisions.
   - If the relevant provision is unavailable, explicitly state
     that the retrieved context does not contain sufficient
     information.
   - Never conclude that a section does not exist in the IPC
     merely because it was not retrieved.

4. PROVISION-SPECIFIC ACCURACY
   - Distinguish between the definition of an offence,
     its essential ingredients, punishment, illustrations,
     special circumstances and exceptions.
   - For theft:
     Section 378 defines theft.
     Section 379 provides punishment for theft.
   - For robbery:
     Section 390 defines robbery.
     Section 392 provides punishment for robbery.
     Section 390 contains conditions involving theft and
     extortion. Explain the applicable conditions when
     the user asks for the complete definition.
   - These examples clarify the relationship between provisions.
     They must not restrict the chatbot to these offences.
   - Apply the same accuracy and verification principles
     to every IPC provision.

5. RETRIEVED CONTEXT
   - Use only retrieved chunks relevant to the user's question.
   - Ignore unrelated chunks even if they contain similar
     keywords or section references.
   - Do not assume that retrieved chunks contain the complete
     text of a section.
   - Use multiple relevant chunks when necessary to construct
     an accurate answer.
   - Consider section continuations across PDF pages.
   - Do not omit important conditions or qualifications
     present in the relevant retrieved text.
   - If the context is insufficient, clearly state what
     information is unavailable.
   - Never fill missing information using assumptions.

6. SOURCE AND PAGE REFERENCES
   - Include original PDF page numbers for factual information.
   - Associate page references with the information they support.
   - If a provision spans multiple pages, include all relevant
     page numbers available in the retrieved metadata.
   - Do not invent page numbers.

7. HISTORICAL DOCUMENT DISCLAIMER
   - The supplied document represents the historical IPC.
   - Do not present it as a complete statement of current
     Indian criminal law.
   - Do not silently replace historical IPC provisions
     with provisions from newer legislation.
   - If asked about current law, explain that the supplied
     historical document alone cannot establish the current
     legal position.

8. PREVIOUS CONVERSATION
   - Use conversation history only to understand follow-up
     questions, references and conversational context.
   - Never treat previous AI-generated responses as evidence.
   - Retrieved document context remains the sole source
     of legal facts.
   - If a follow-up question refers to an earlier answer,
     identify the relevant subject but verify all legal facts
     against the retrieved context.

9. RESPONSE STRUCTURE
   - Start directly with the answer.
   - Choose headings and formatting according to the
     actual question.
   - Use concise paragraphs for straightforward questions.
   - Use bullet points or numbered lists when explaining
     multiple ingredients, conditions or steps.
   - Use tables only when a table genuinely helps answer
     a question involving comparison or structured data.
   - Do not force a fixed template for every answer.
   - Do not automatically generate headings such as
     "Key Differences", "Related Offences" or "Punishment".
   - Avoid unnecessary introductions, conclusions and repetition.
   - Keep answers clear, relevant and appropriately detailed.

10. MARKDOWN FORMATTING
    - Use valid Markdown.
    - Do not generate HTML tags such as <br>.
    - Use headings only when they improve readability.
    - Do not put lengthy statutory definitions or explanations
      inside Markdown tables.
    - Ensure tables have consistent columns and rows.
    - Do not repeat information under multiple headings.

11. SOURCE GROUNDING
    - Every factual legal claim must be supported by
      the retrieved document context.
    - Clearly distinguish direct statements from reasonable
      inferences.
    - Never present an inference as a direct statement
      from the IPC.
    - If information is missing, explicitly state:
      "Not available in the retrieved document context."
    - Prefer accuracy over completeness.

12. ANSWER QUALITY
    - Answer the exact question before adding any explanation.
    - Do not introduce information merely because it appears
      in the retrieved context.
    - Do not generate unrelated comparisons or examples.
    - Do not repeat the same information.
    - Do not fabricate missing information.
    - Keep the response focused without making it
      unnecessarily short or excessively long.
    - The answer must be understandable without requiring
      the user to interpret the statutory text themselves.

Previous conversation:
{history}

Retrieved IPC context:
{context}

User question:
{question}

Generate one accurate, relevant and well-structured answer.
""")



# STEP 8: Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# STEP 9: Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# STEP 10: Accept user question
user_question = st.chat_input("Ask a question about the IPC...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    answer = ""
    retrieved_docs = []

    with st.chat_message("assistant"):
        with st.spinner("Searching IPC document..."):
            try:
                # STEP 11: Retrieve relevant documents
                retrieved_docs = retriever.invoke(user_question)

                # DEBUG: Inspect retrieved documents
                print("\n========== RETRIEVAL DEBUG ==========")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\nCHUNK {i + 1}")
                    print("PAGE:", doc.metadata.get("page", "Unknown"))
                    print("CONTENT:")
                    print(doc.page_content)
                    print("-" * 60)
                print("=====================================\n")

                # STEP 12: Prepare context
                context_parts = []
                for doc in retrieved_docs:
                    page = doc.metadata.get("page", "Unknown")
                    if isinstance(page, int):
                        page += 1
                    context_parts.append(
                        f"Source page: {page}\n{doc.page_content}"
                    )

                context = "\n\n".join(context_parts)

                if not context:
                    raise ValueError("No relevant IPC context was retrieved.")

                # STEP 13: Prepare conversation history
                history = ""
                for msg in st.session_state.messages[-7:-1]:
                    history += f"{msg['role']}: {msg['content']}\n"

                
                # STEP 14: Generate answer
                formatted_messages = prompt.format_messages(
                    history=history,
                    context=context,
                    question=user_question
                )

                response = llm.invoke(formatted_messages)

                print("\n========== GROQ RESPONSE DEBUG ==========")
                print("Response:", response)
                print("Usage:", response.usage_metadata)
                print("Additional Info:", response.additional_kwargs)
                print("=========================================\n")

                answer = response.content

                if isinstance(answer, list):
                    answer = "\n".join(
                        str(item.get("text", ""))
                        if isinstance(item, dict)
                        else str(item)
                        for item in answer
                    )

                answer = str(answer).strip()

                if not answer:
                    answer = (
                        "The model could not generate a complete answer. "
                        "Please try asking the question again."
                    )

                    raise ValueError(
                        f"Groq returned an empty response. "
                        f"Finish reason: {finish_reason}. "
                        "Check terminal logs."
                    )

                # STEP 15: Clean response formatting
                answer = re.sub(
                    r"<br\s*/?>",
                    "\n",
                    answer,
                    flags=re.IGNORECASE
                )

                answer = html.unescape(answer)

                # STEP 16: Display answer once
                st.markdown(answer)


                # STEP 17: Display retrieved sources
                if retrieved_docs:
                    st.subheader("Retrieved Sources")
                    source_pages = set()

                    for doc in retrieved_docs:
                        page = doc.metadata.get("page", "Unknown")
                        if isinstance(page, int):
                            page += 1
                        source_pages.add(page)

                    for page in sorted(
                        source_pages,
                        key=lambda x: (isinstance(x, str), x)
                    ):
                        st.caption(f"IPC PDF - Page {page}")

            except Exception as e:
                st.warning("No relevant IPC document chunks were retrieved.")

    # STEP 18: Save assistant response ONCE
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# STEP 19: Sidebar
with st.sidebar:
    st.header("About the Chatbot")
    st.write(
        "This chatbot uses Retrieval-Augmented Generation (RAG) "
        "to answer questions from the uploaded IPC document."
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.warning(
        "Educational project only. Not a substitute for professional legal advice."
    )
