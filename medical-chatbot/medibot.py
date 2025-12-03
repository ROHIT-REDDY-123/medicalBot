
import os
import streamlit as st
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # ✅ FIXED: Ollama instead of broken HF

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    """Load Ollama LLM (LOCAL - FAST - RELIABLE)"""
    try:
        llm = OllamaLLM(
            model="phi3:mini",  # ✅ ollama pull phi3:mini
            temperature=0.1,
            num_predict=200
        )
        st.success("✅ Ollama LLM loaded!")
        return llm
    except Exception as e:
        st.error(f"❌ Ollama error: {e}")
        st.info("💡 Run: `ollama serve` and `ollama pull phi3:mini`")
        st.stop()

def main():
    st.set_page_config(page_title="MediBot", page_icon="🏥")
    st.title("🏥 **MediBot** - Medical Assistant")
    st.markdown("---")


    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('assistant'):
            with st.spinner("🔍 Searching medical database..."):
                try:
                    # Load vectorstore
                    vectorstore = get_vectorstore()
                    
                    # Load LLM (Ollama primary, Groq fallback)
                    llm = load_llm()
                    
                    # Custom medical prompt
                    CUSTOM_PROMPT_TEMPLATE = """
                    You are a medical assistant. Answer using ONLY the provided medical context.
                    If the answer is not found in context, say: "I don't have that medical information."

                    MEDICAL CONTEXT: {context}

                    PATIENT QUESTION: {question}

                    MEDICAL RESPONSE:"""

                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )

                    # Get response
                    start_time = time.time()
                    response = qa_chain.invoke({'query': prompt})
                    elapsed = time.time() - start_time

                    result = response["result"].strip()
                    sources = response["source_documents"]

                    # Format response with sources
                    source_info = "\n\n**📚 Sources:**\n"
                    for i, doc in enumerate(sources[:2], 1):
                        page = doc.metadata.get('page', 'N/A')
                        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        snippet = doc.page_content[:100] + "..."
                        source_info += f"{i}. **{source} (Page {page})**: {snippet}\n"

                    full_response = f"{result}\n\n{source_info}"
                    
                    st.markdown(full_response)
                    st.caption(f"⏱️ Response time: {elapsed:.1f}s")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("""
                    **Quick Fix:**
                    1. Terminal: `ollama serve` (keep running)
                    2. New Terminal: `ollama pull phi3:mini`
                    3. Refresh page
                    """)

        # Add to chat history
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': full_response
        })

if __name__ == "__main__":
    main()
