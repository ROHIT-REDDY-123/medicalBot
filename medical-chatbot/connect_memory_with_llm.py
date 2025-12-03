import os
import time
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


# ----------------------------
# Ollama Setup
# ----------------------------
MODEL_ID = "phi3:mini"

llm = OllamaLLM(
    model=MODEL_ID,
    temperature=0,
    num_predict=150
)

# ----------------------------
# Medical RAG Prompt
# ----------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant. Answer using ONLY the provided medical context.
If the answer is not in the context, say "I don't have that medical information available."

CONTEXT: {context}

QUESTION: {question}

ANSWER:
"""

prompt_template = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ----------------------------
# Load FAISS DB
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# ----------------------------
# Build QA Chain
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# ----------------------------
# Consultation Loop
# ----------------------------
while True:
    user_query = input("\nMedical Question (type 'quit' to exit): ").strip()

    if user_query.lower() in ['quit', 'exit', 'q']:
        break
    if not user_query:
        print("Please enter a question.")
        continue

    start_time = time.time()

    try:
        response = qa_chain.invoke({"query": user_query})
        elapsed = time.time() - start_time

        print(f"\nAnswer ({elapsed:.1f}s):")
        print(response["result"].strip())

        print("\nSources:")
        for i, doc in enumerate(response["source_documents"][:2], 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'N/A')
            snippet = doc.page_content[:120]
            print(f"{i}. {source} (Page {page}) - {snippet}...")

    except Exception as e:
        print(f"Error: {str(e)}")


print("\nMedical Assistant stopped.")
