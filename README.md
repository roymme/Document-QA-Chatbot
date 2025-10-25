
# 🧠 Document-QA-Chatbot

An **AI-powered Question-Answering (QA) system** built using **LangChain**, **FAISS**, and **Google Gemini (Generative AI)**.  
This chatbot allows users to upload documents (PDFs, text files, etc.) and ask context-aware questions directly from their content.


---

## 🚀 Features

- 📄 Upload and process multiple documents  
- 🧩 Chunk text and create vector embeddings  
- 🔍 Retrieve context using **FAISS vector store**  
- 💬 Generate intelligent answers using **Google Gemini API**  
- 🌐 Deployed seamlessly on **Streamlit Cloud**  
- 🔒 Secure API key management using `st.secrets`

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | LangChain + Python |
| Vector Store | FAISS |
| Embedding Model | Google Generative AI Embeddings |
| LLM | Gemini 1.5 (via `ChatGoogleGenerativeAI`) |
| Deployment | Streamlit Cloud |

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Document-QA-Chatbot.git
cd Document-QA-Chatbot
````

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Setting up the Google API Key

### Option 1 — Local Development

Create a file named `.streamlit/secrets.toml` inside your project root folder:

```
.streamlit/
└── secrets.toml
```

Add the following content:

```toml
[google]
api_key = "YOUR_GOOGLE_API_KEY_HERE"
```

### Option 2 — On Streamlit Cloud

1. Go to your deployed app’s **Settings → Edit secrets**
2. Paste the same TOML content above.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 💡 Example Usage

1. Upload a document (PDF or text file).
2. The app extracts and embeds text chunks.
3. Ask a question — for example:

   > “Summarize this document.”
   > “What are the main findings mentioned in section 3?”
4. The chatbot responds contextually using the retrieved embeddings and Gemini model.

---

## 🧩 File Structure

```
Document-QA-Chatbot/
│
├── app.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
└── pdf_files/
    └── sample.pdf
```

---

## ⚙️ Environment Variables

| Variable         | Description                                |
| ---------------- | ------------------------------------------ |
| `GOOGLE_API_KEY` | Your Gemini API key stored in `st.secrets` |

---


## 🧑‍💻 Author

**Shashwata Roy**
🎓 B.Tech, Metallurgical and Material Engineering — Jadavpur University
💼 aspiring GenAI Engineer
📧 [shashwataroy17@gmail.com](mailto:shashwataroy17@gmail.com)
🌐 [LinkedIn](https://www.linkedin.com/in/shashwata-roy17)

---

## 📝 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it with attribution.

---

## ⭐ Acknowledgements

* [LangChain Documentation](https://python.langchain.com/)
* [Google Generative AI](https://ai.google.dev/)
* [Streamlit Docs](https://docs.streamlit.io/)
* [FAISS by Facebook AI](https://faiss.ai/)

---

### ❤️ Support

If you like this project, please ⭐ star the repository and share it with others!

```

