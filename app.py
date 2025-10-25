import os
import operator
from typing import TypedDict, Annotated, Sequence

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.embeddings import SentenceTransformerEmbeddings

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PDF_DIRECTORY = r"pdf_files"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="🤖",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "app" not in st.session_state:
    st.session_state.app = None
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "abc123"}}
if "initialized" not in st.session_state:
    st.session_state.initialized = False

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    context: str

def load_and_process_documents(pdf_directory: str):
    with st.spinner("Loading PDF documents..."):
        dir_loader = DirectoryLoader(
            pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        documents = dir_loader.load()
        if not documents:
            st.error("No PDF documents found in the specified directory!")
            return None
        st.info(f"Loaded {len(documents)} pages from PDF files")
    with st.spinner("Processing documents and creating embeddings..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        st.info(f"Created {len(splits)} text chunks")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.success("✅ Document processing complete!")
        return vectorstore

def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Instructions:
- Answer questions accurately based on the provided context , it its not on the context say you don't know
- If you cannot find the answer in the context, say so clearly
- Maintain conversation history and reference previous questions when relevant
- Be conversational and helpful
- Do not mention about being an AI model
- Cite specific information from the documents when possible such as pg number,etc

Previous conversation:
{chat_history}

Current question: {input}

Answer:"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def create_langgraph_app(rag_chain):
    def retrieve_context(state: AgentState):
        last_message = state["messages"][-1]
        chat_history = []
        for msg in state["messages"][:-1]:
            if isinstance(msg, HumanMessage):
                chat_history.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                chat_history.append(("ai", msg.content))
        response = rag_chain.invoke({
            "input": last_message.content,
            "chat_history": chat_history
        })
        return {
            "messages": [AIMessage(content=response["answer"])],
            "context": "\n\n".join([doc.page_content for doc in response["context"]])
        }
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_and_answer", retrieve_context)
    workflow.set_entry_point("retrieve_and_answer")
    workflow.add_edge("retrieve_and_answer", END)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

if not st.session_state.initialized:
    if os.path.exists(PDF_DIRECTORY):
        with st.spinner("🔄 Loading PDF documents..."):
            vectorstore = load_and_process_documents(PDF_DIRECTORY)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                rag_chain = create_rag_chain(vectorstore)
                st.session_state.app = create_langgraph_app(rag_chain)
                st.session_state.initialized = True
    else:
        st.error(f"PDF directory not found: {PDF_DIRECTORY}")
        st.stop()

with st.sidebar:
    st.title("🤖 Document QA Chatbot")
    st.divider()
    st.subheader("📊 Status")
    if st.session_state.vectorstore:
        st.success("✅ PDFs loaded and ready")
    else:
        st.warning("⚠️ No PDFs loaded")
    st.info(f"💬 Messages: {len(st.session_state.messages)}")
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.config = {"configurable": {"thread_id": f"abc{len(st.session_state.messages)}"}}
        st.rerun()
    st.divider()

st.title("🤖 Document QA Chatbot")
st.caption("How can i help you today ? ")

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question ..."):
    if not st.session_state.app:
        st.error("⚠️ System not armed . ")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                graph_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        graph_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        graph_messages.append(AIMessage(content=msg["content"]))
                result = st.session_state.app.invoke(
                    {"messages": graph_messages},
                    config=st.session_state.config
                )
                response = result["messages"][-1].content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()
