# Generative AI Projects

Welcome to my Generative AI Projects portfolio! This repository showcases a variety of projects exploring modern generative AI techniques including Retrieval-Augmented Generation (RAG), document-based Q&A systems, LLM-driven chat interfaces, and educational tools using OpenAI APIs and local language models.

Each folder represents a standalone project with its own codebase, and supporting assets (e.g., notebooks, APIs, Streamlit apps, and embeddings).

---

## Project Directory

```code
generative-ai-projects/
├── chatdocs-openai-faiss/
│   ├── generate_embeddings_faiss.ipynb
│   ├── app_chatdocs_streamlit.py
│   ├── index.faiss
├── ai-tutor-for-children/
│   ├── AI_Tutor_for_Children_Presentation.pptx
├── llms-in-financial-market-research/
│   ├── llms-in-financial-market-research.pdf
└── README.md                 # Main project documentation
```

---

## Project Highlights

### 1. `chatdocs-openai-faiss`
- **Type:** Document Q&A Assistant (RAG)
- **Tech:** OpenAI `text-embedding-ada-002`, FAISS, Streamlit
- **Function:** Converts research papers into searchable embeddings, enabling question-answering via GPT.
- **Deployment:** Local + Streamlit Cloud

### 2.  `ai-tutor-for-children`
- **Type:** Interactive AI-based tutor for kids
- **Tech:** Ollama + local audio pipeline + streaming interface (planned with Simli)
- **Function:** Children can ask questions by voice; responses are generated using LLM and planned to be spoken aloud.

### 3.  `llms-in-financial-market-research`
- **Type:** Research paper
- **Focus:** Applications of LLMs like GPT, BERT, and BloombergGPT in financial forecasting, text classification, QA, and summarization.
- **Assets:** Academic-style report exploring datasets like FiQA, FinNER, and FinQA.

---

##  Tools & Technologies

- **LLMs & APIs:** OpenAI GPT-4, Ollama, FinGPT, BERT, FAISS
- **Frameworks:** Streamlit, Python, PyPDF2, dotenv
- **Techniques:** Retrieval-Augmented Generation, Embedding Search, Sentiment Analysis, QA over unstructured data
- **Libraries:** LangChain (planned), Transformers, tqdm

---

