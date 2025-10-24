import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational PDF Chatbot")
st.write("Upload a PDF document and ask questions about its content. The app will retrieve relevant information from the document and provide answers based on the context.")

api_key=st.text_input("Enter your Groq API Key:", type="password")
if api_key:
    model_name=st.sidebar.selectbox("Select LLM Model",[ "groq/compound","meta-llama/llama-4-scout-17b-16e-instruct","openai/gpt-oss-120b"]  )
    llm=ChatGroq(groq_api_key=api_key,model_name=model_name)
    session_id=st.text_input("Session_ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            
            loader= PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

            text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
            splits=text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(splits, embeddings)
            retriever=vectorstore.as_retriever()

            contextualize_q_system_prompt=(
                "Given a Chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question that can be understood "
                "without the chat history. Do not answer the question,"
                "just reformulate it if needed and otherwise return it as is."

            )
            contextualize_q_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system",contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}"),
                ]
            )
            history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

            system_prompt=(
                "You are a helpful AI assistant that helps people find information "
                "in long documents. Use the following pieces of context to answer "
                "the users question. If you don't know the answer, just say that "
                "you don't know, don't try to make up an answer. Always be "
                "truthful and never fabricate information. If the question is not "
                "about the context, politely inform them that you are tuned to "
                "only answer questions that are related to the context."
                "\n\n{context}\n\n"
            )
            qa_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system",system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}"),
                ]
            )
            question_answer_chain=create_stuff_documents_chain(llm,prompt=qa_prompt)
            rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            user_input=st.text_input("Ask a question about the document:")
            if user_input:
                session_history=get_session_history(session_id)
                response=conversational_rag_chain.invoke({"input":user_input},
                config={
                    "configurable":{"session_id":session_id,"model_name":model_name,"file_name":file_name}
                }
                )
                st.write(st.session_state.store)
                st.write("Assistant:",response["answer"])
                st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter your Groq API Key to use the application.")
            
