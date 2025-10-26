import os
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Load the API Key
os.environ["OPENAI_API_KEY"] = os.getenv("openapi_key3")

# Initialize the LLM
llm = OpenAI(temperature=0.7, max_tokens=500)

st.title("URL Research Assistant")

st.sidebar.title("Please enter the URL you want to research")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
urls = [url for url in [url1, url2, url3] if url]

# Initialize session state -  so we dont need to reload the every time
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = OpenAI(temperature=0.9, max_tokens=500)

# Empty placeholder for status updates
status_placeholder = st.empty()

# Creating process button to start the research
if st.sidebar.button("Start Research"):

    if not urls:
        st.sidebar.error("Please provide at least one URL.")
    else:
        # loading data from the urls
        loaders = UnstructuredURLLoader(urls=urls)
        status_placeholder.info("Loading data from the provided URLs...")
        data = loaders.load()

        # Split the data and create chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        status_placeholder.info("Splitting the data into chunks...")
        docs = text_splitter.split_documents(data)

        # Create the embeddings for these chunk and save using FAISS index
        embeddings = OpenAIEmbeddings()
        status_placeholder.info("Creating embeddings and building the vector index...")
        vector_index_openai = FAISS.from_documents(docs, embedding=embeddings)

        # Storing embeddings on the pickle file
        vector_index_openai.save_local("faiss_index")

        # Load the embeddings
        vector_index = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        st.session_state.vector_index = vector_index
        status_placeholder.info("Research data has been processed")
        time.sleep(1)
        status_placeholder.success("Vector index is ready for querying.")
        time.sleep(2)
        status_placeholder.empty()
        time.sleep(1)
        st.subheader("Ask questions about the researched URLs")


# Display the existing messages
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])


if st.session_state.vector_index is not None:

    # Input box for user question
    if user_question := st.chat_input("Enter your question about the provided URLs:"):
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Retrieve the similar embeddings for a question and call the final LLM for the final answer
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=st.session_state.vector_index.as_retriever()
        )
        response = chain({"question": user_question}, return_only_outputs=True)
        print(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"{response['answer']}\n\nSources:\n{response['sources']}",
            }
        )
        # Show assistant message immediately
        with st.chat_message("assistant"):
            st.markdown(f"{response['answer']}\n\nSources:\n{response['sources']}")
