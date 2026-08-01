# NHPT Heritage Assistant

A computer vision and conversational AI system that identifies architectural styles from photographs and provides visitors with grounded, source-cited conservation guidance through a retrieval-augmented chatbot.

Built for the National Heritage Preservation Trust (NHPT) coursework project — Machine Learning and Related Applications (NB627BSDS).

## What it does

1. A visitor uploads a photo of a building through the Streamlit web app.
2. A fine-tuned ResNet50 model classifies the architectural style (8 classes) and returns a confidence score.
3. The predicted style is turned into a question and passed to a LangChain RAG pipeline.
4. The pipeline retrieves relevant chunks from a purpose-written heritage knowledge base, sends them to an LLM (Groq API), and returns a grounded, cited answer — with explicit hedging language if the CV model's confidence is low.
5. Visitors can continue the conversation naturally; the chatbot remembers earlier turns.

## Features

- **Image classification** — ResNet50 transfer learning model, 94.3% top-1 accuracy across 8 architectural styles (Achaemenid, American Foursquare, American Craftsman, Ancient Egyptian, Art Deco, Art Nouveau, Baroque, Bauhaus)
- **Grad-CAM visualizations** for model interpretability
- **RAG chatbot** grounded in an 8-document heritage knowledge base, with source citations
- **Conversation memory** — handles natural follow-up questions across turns
- **Confidence-aware responses** — the chatbot hedges when the CV model is uncertain, rather than stating a possibly-wrong classification as fact
- **Streamlit web app** — image upload, live classification, and chat in one interface

## Project structure

```
.
├── app.py                              # Streamlit web app
├── chatbot_core.py                     # Shared RAG + CV logic (used by app.py and the notebook)
├── requirements.txt
├── train_architectural_classifier.ipynb # CV model training, evaluation, Grad-CAM
├── rag_chatbot.ipynb                    # RAG pipeline development notebook
├── knowledge_base/                      # 8 Markdown documents, one per architectural style
├── models/
│   └── resnet_architectural_classifier_v2.keras
├── class_indices_v1.json                # Maps model output indices to style names
├── chroma_db/                           # Persisted vector store (generated, not hand-edited)
└── example_conversations.json           # Saved sample conversations, including CV integration
```

## Setup

### 1. Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. LLM API key (Groq)

This project uses [Groq](https://console.groq.com) to run the LLM (currently `openai/gpt-oss-20b`) — free, cloud-hosted, no local GPU or large downloads required.

1. Create a free account at console.groq.com and generate an API key.
2. Set it as an environment variable before running anything:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your-key-here"

# macOS/Linux
export GROQ_API_KEY="your-key-here"
```

To avoid setting this every session, save it permanently (Windows):
```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your-key-here", "User")
```

**Never commit your API key or `.streamlit/secrets.toml` to this repository.**

### 3. Train the CV model (optional — a trained model is already included)

Open `train_architectural_classifier.ipynb`, update `DATA_DIR` to point at your local copy of the architectural styles dataset, and run all cells. This regenerates `models/resnet_architectural_classifier_v2.keras` and `class_indices_v1.json`.

### 4. Build the knowledge base vector store (first run only)

Run through `rag_chatbot.ipynb` cells 1–4, or let `app.py` build it automatically on first launch if `chroma_db/` doesn't exist yet.

### 5. Run the app

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`. Upload a photo in the sidebar to get a style classification, or type a question directly into the chat.

## Model performance summary

| Model | Top-1 test accuracy | Notes |
|---|---|---|
| Baseline CNN (scratch) | ~42.8% val | Confirmed transfer learning was necessary |
| MobileNetV2 (transfer learning) | ~86.8% | Faster, smaller, used only as a comparison baseline |
| **ResNet50 (transfer learning)** | **94.3%** | **Selected model — used in the deployed app** |

Known limitation: the model confuses American Foursquare and American Craftsman styles in some cases, reflecting genuine architectural similarity between the two — see `train_architectural_classifier.ipynb` for the full confusion matrix and analysis.

## Tech stack

- **CV**: TensorFlow/Keras, ResNet50 (ImageNet pretrained), Grad-CAM
- **RAG**: LangChain, Chroma (vector store), `sentence-transformers/all-MiniLM-L6-v2` (local embeddings)
- **LLM**: Groq API (`openai/gpt-oss-20b`)
- **App**: Streamlit

## Notes

- All CV training was done on CPU-only hardware, which shaped several architecture and hyperparameter choices — see the accompanying technical report for details.
- The RAG pipeline uses a hybrid retrieval strategy (keyword-triggered full-document retrieval for named styles, semantic search as a fallback) rather than pure semantic search, after testing showed pure similarity ranking was unreliable at this knowledge base's small scale.
- `example_conversations.json` contains real, unedited transcripts demonstrating high-confidence classification, low-confidence hedging, multi-turn memory, and general knowledge-base Q&A.

## License

Coursework project — NIBM / The City University, BSc Data Science, Machine Learning and Related Applications (NB627BSDS).
