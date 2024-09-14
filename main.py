import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Initialize Streamlit App
st.title("RAG QA bot by Gemini-1.5")

# Load PDF and process it
loader = PyPDFLoader("yolov9_paper.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Setup Chroma Vectorstore
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize session state for prompt history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Display the last five prompts
st.subheader("Chat History")
if st.session_state["history"]:
    for i, q in enumerate(st.session_state["history"][-5:], 1):  # Show only the last 5 prompts
        st.write(f"{i}. {q}")

# Accept user input
query = st.chat_input("Say something: ")

# Process the query
if query:
    # Add the query to the session state history
    st.session_state["history"].append(query)

    # Create chain for question-answering
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get response
    response = rag_chain.invoke({"input": query})
    
    # Display the response
    st.write(response["answer"])

