import gradio as gr
import torch
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load dataset
dataset = load_dataset("MedRAG/textbooks", split="train[:10%]")
texts = dataset["content"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load LLM
tokenizer, model = None, None
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Function to retrieve context
def retrieve_context(query, k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)
    return " ".join([texts[i] for i in indices[0]])

# Function to generate response
def generate_response(question):
    if model is None or tokenizer is None:
        load_model()  # Load model only when needed

    context = retrieve_context(question)
    prompt = f"Context: {context} \nQuestion: {question} \nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Ask a medical question"),
    outputs=gr.Textbox(label="Answer"),
    title="Medical RAG AI"
)

# Launch the app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
