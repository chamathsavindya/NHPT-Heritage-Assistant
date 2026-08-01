import streamlit as st
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from chatbot_core import load_cv_model, classify_image, cv_result_to_question, ask_with_memory

CV_MODEL_PATH = "models/resnet_architectural_classifier_v2.keras"
CLASS_INDICES_PATH = "class indices/resnet_class_indices_v2.json"
CHROMA_DIR = "chroma_db"

st.set_page_config(page_title="NHPT Heritage Assistant", page_icon="🏛️", layout="wide")

@st.cache_resource
@st.cache_resource
def load_resources():
    print("Loading CV model...")
    cv_model, class_indices = load_cv_model(CV_MODEL_PATH, CLASS_INDICES_PATH)
    print("CV model loaded.")

    print("Loading embeddings + Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    print("Chroma loaded.")

    print("Connecting to Groq...")
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
    print("Groq connected.")

    return cv_model, class_indices, vectordb, llm
cv_model, class_indices, vectordb, llm = load_resources()

st.title("🏛️ NHPT Heritage Assistant")
st.caption("Upload a photo of a building feature, or just ask a question about architectural heritage.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📷 Identify a Building")
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze this image"):
            with st.spinner("Classifying..."):
                cv_result = classify_image(image, cv_model, class_indices)

            st.success(f"**Predicted style:** {cv_result['predicted_style']}")
            st.write(f"**Confidence:** {cv_result['confidence']:.1%}")
            with st.expander("Top 3 predictions"):
                for p in cv_result["top_predictions"]:
                    st.write(f"- {p['style']}: {p['confidence']:.1%}")

            question = cv_result_to_question(cv_result)
            with st.spinner("Getting more information..."):
                answer, sources = ask_with_memory(question, st.session_state.chat_history, vectordb, llm)

            st.session_state.chat_history.append((question, answer))
            st.session_state.messages.append({"role": "user", "content": f"📷 *Uploaded a photo — identified as {cv_result['predicted_style']} ({cv_result['confidence']:.0%} confidence)*"})
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

user_question = st.chat_input("Ask about an architectural style...")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_with_memory(user_question, st.session_state.chat_history, vectordb, llm)
        st.write(answer)
        st.caption(f"Sources: {', '.join(sources)}")
    st.session_state.chat_history.append((user_question, answer))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})