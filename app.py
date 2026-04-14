import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title="Offline RAG Chatbot", layout="wide")
st.title("📄 Offline RAG PDF Chatbot (Ollama)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded!")

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    st.info(f"Chunks: {len(docs)}")

    # Embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Local LLM (Ollama)
    llm = Ollama(model="llama3")

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            prompt = ChatPromptTemplate.from_template(
                """Answer using ONLY the context below:
                {context}

                Question: {input}
                """
            )

            qa_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, qa_chain)

            response = rag_chain.invoke({"input": query})

            st.subheader("Answer:")
            st.write(response["answer"])