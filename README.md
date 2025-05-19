# Innovatech Solutions RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot powered by Google Gemini Flash and LangChain. This app lets you upload knowledge bases in TXT, PDF, or JSON formats and interactively query them.

---

## Features

* **Multi-format support**: Ingest plain text (`.txt`), PDF (`.pdf`), or JSON (`.json`) files as knowledge bases.
* **RAG pipeline**: Uses LangChain with Google Gemini Flash (`gemini-1.5-flash-latest`) for embeddings and chat.
* **Streamlit UI**: Simple web interface with sidebar controls for chunk size and chunk overlap.
* **Caching**: Pipeline initialization is cached to speed up repeated runs.
* **Chat logging**: Conversations are logged to `chat_log.txt` with timestamps.

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/innovatech-rag-chatbot.git
   cd innovatech-rag-chatbot
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv env
   source env/bin/activate    # macOS/Linux
   env\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   * Create a `.env` file in the project root:

     ```bash
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

5. **(Optional) Upgrade pip**

   ```bash
   pip install --upgrade pip
   ```

---

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

* Open the provided URL (usually `http://localhost:8501`).
* Adjust **Chunk Size** and **Chunk Overlap** in the sidebar if needed.
* Upload your knowledge base (TXT, PDF, or JSON).
* Enter questions in the chat box and click **Send**.
* View and scroll through chat history in the main panel.

Conversations are automatically logged to `chat_log.txt`.

---

## Example JSON Knowledge Base

```json
[
  {
    "section": "Company Overview",
    "content": "Innovatech Solutions is a leading technology company founded in 2010..."
  },
  {
    "section": "Services",
    "content": "1. Custom AI Model Development..."
  }
]
```

Save as `kb.json` and upload via the UI.

---

## Deployment to Streamlit Cloud

1. Push your code to a GitHub repo.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and create a new app.
3. Connect your GitHub repository and set the main file to `streamlit_app.py`.
4. In the app settings, add `GEMINI_API_KEY` under **Secrets**.
5. Deploy and share the live URL.

---

## Next Steps

* **UI enhancements**: Dark mode, message timestamps, export chat history.
* **Additional formats**: Support for DOCX, CSV.
* **Database logging**: Store conversations in a database instead of text file.
* **Authentication**: Add user login or API key protection.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
