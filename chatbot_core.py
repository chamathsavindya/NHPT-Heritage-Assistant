"""
Shared RAG logic used by both the notebook and the Streamlit app.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

STYLE_TO_FILE = {
    "achaemenid": "achaemenid_architecture.md",
    "foursquare": "american_foursquare_architecture.md",
    "craftsman": "american_craftsman_style.md",
    "egyptian": "ancient_egyptian_architecture.md",
    "art deco": "art_deco_architecture.md",
    "art nouveau": "art_nouveau_architecture.md",
    "baroque": "baroque_architecture.md",
    "bauhaus": "bauhaus_architecture.md",
}

CONFIDENCE_THRESHOLD = 0.55


def load_cv_model(model_path, class_indices_path):
    model = tf.keras.models.load_model(model_path, custom_objects={"preprocess_input": preprocess_input})
    dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")
    _ = model.predict(dummy_input, verbose=0)
    with open(class_indices_path) as f:
        class_indices = json.load(f)
    return model, class_indices


def classify_image(pil_image, model, class_indices):
    img = pil_image.convert("RGB").resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_batch, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:3]
    top_predictions = [{"style": class_indices[str(i)], "confidence": float(preds[i])} for i in top_idx]
    return {
        "predicted_style": top_predictions[0]["style"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }


def cv_result_to_question(cv_result):
    style = cv_result["predicted_style"]
    confidence = cv_result["confidence"]
    if confidence < CONFIDENCE_THRESHOLD:
        alt = cv_result["top_predictions"][1]["style"]
        note = (
            f"[Note: our vision system is not fully confident — it estimates {confidence:.0%} "
            f"likelihood this is {style}, with {alt} as a possible alternative. "
            f"Please mention this uncertainty in your answer rather than stating the style as fact.] "
        )
    else:
        note = f"[Our vision system identified this with {confidence:.0%} confidence.] "
    return (
        f"{note}A visitor photographed a building feature identified as {style}. "
        f"What should they know about this style and what to look out for regarding its condition?"
    )


def detect_styles(text):
    text_lower = text.lower()
    matched = []
    for keyword, filename in STYLE_TO_FILE.items():
        if keyword in text_lower and filename not in matched:
            matched.append(filename)
    return matched


def condense_question(question, history, llm):
    if not history:
        return question
    history_text = "\n".join([f"Visitor: {q}\nAssistant: {a}" for q, a in history])
    prompt = f"""Given this conversation history and a follow-up question, rewrite the
follow-up as a standalone question that includes any implied context from the history.
Only output the rewritten question, nothing else.

HISTORY:
{history_text}

FOLLOW-UP QUESTION: {question}

STANDALONE QUESTION:"""
    return llm.invoke(prompt).content.strip()


def get_chunks_for_files(vectordb, matched_files):
    all_text, all_meta = [], []
    for filename in matched_files:
        source_path = f"knowledge_base\\{filename}"
        raw = vectordb.get(where={"source": source_path}, include=["documents", "metadatas"])
        all_text.extend(raw["documents"])
        all_meta.extend(raw["metadatas"])
    return all_text, all_meta


def ask_with_memory(question, history, vectordb, llm):
    search_question = condense_question(question, history, llm)
    matched_files = detect_styles(search_question)

    if matched_files:
        results_text, results_meta = get_chunks_for_files(vectordb, matched_files)
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_question)
        results_text = [d.page_content for d in docs]
        results_meta = [d.metadata for d in docs]

    context = "\n\n".join([f"[Source: {m['source']}]\n{t}" for t, m in zip(results_text, results_meta)])
    sources = sorted(set(m["source"] for m in results_meta))
    history_text = "\n".join([f"Visitor: {q}\nAssistant: {a}" for q, a in history])

    prompt = f"""You are the NHPT Heritage Assistant, helping visitors understand historic sites.

Rules:
1. Answer ONLY using the CONTEXT below. If the context doesn't contain the answer, say
   "I don't have that information in NHPT's records" rather than guessing.
2. Each context chunk is labeled with its source file. Only use chunks relevant to the
   specific building style(s) being asked about.
3. Use the CONVERSATION HISTORY to understand follow-up questions.
4. Keep answers concise (2-4 sentences) and visitor-friendly.

CONVERSATION HISTORY:
{history_text if history_text else "(no previous messages)"}

CONTEXT:
{context}

VISITOR QUESTION: {question}

ANSWER:"""

    response = llm.invoke(prompt).content.strip()
    return response, sources