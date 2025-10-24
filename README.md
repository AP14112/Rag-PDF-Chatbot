# Conversational PDF Chatbot

This is a smart little app that lets you upload your PDF documents and have a conversation about them. It's built with **Streamlit** for the interface and uses **LangChain**, **Groq**, and **ChromaDB** to understand your documents and answer questions.

The best part? It **remembers your chat history**, so you can ask follow-up questions just like you're talking to a person.

## Key Features

* **Chat with your PDFs:** Upload one or more PDF files to create a knowledge base.
* **Chat with past context** The app keeps track of your conversation, so you can ask follow-up questions (e.g., "What did you say about that last point?").
* **Session-Based Memory:** Uses a **Session ID** to keep different conversations separate.
* **Super-Fast Answers:** Powered by the **Groq API** for near-instant responses.
* **Private & Local RAG:** Uses **HuggingFace** embeddings and an in-memory **ChromaDB** vector store to find the *exact* information you need in your docs.

---

## How to Run It

1.  **Clone the Repository**
    ```bash
    git clone [https://your-repository-url.git](https://your-repository-url.git)
    cd your-repository-directory
    ```

2.  **Install Requirements**
    This project has a few dependencies. You can install them all using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install `streamlit`, `langchain-groq`, `langchain-chroma`, `langchain-huggingface`, `pypdf`, `python-dotenv`, etc.)*

3.  **Set Up API Keys**
    Create a `.env` file in the main folder and add your API keys. You'll need a Groq key and a Hugging Face token (for the embeddings).

    ```
    GROQ_API_KEY="gsk_YOUR_REAL_GROQ_KEY"
    HF_API_TOKEN="hf_YOUR_REAL_HUGGINGFACE_TOKEN"
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```
    *(Replace `app.py` with the name of your Python file if it's different.)*

---

## How to Use the App

Once the app is running in your browser:

1.  Paste your **Groq API Key** into the first text box.
2.  Use the file uploader to **upload one or more PDF files**. The app will process them instantly.
3.  (Optional) Change the **Session ID** if you want to start a new, fresh conversation (or return to an old one).
4.  Type your question about the documents into the chat box at the bottom and press Enter.
5.  You  can ask up following questions and the AI will remember what you've already talked about.
