import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data_loader import load_assessments

# Load model ONLY when needed (safe for Render)
model = None
index = None
assessments = None


def init():
    """
    Initialize model + embeddings safely (call once)
    """
    global model, index, assessments

    print("Loading assessments...")
    assessments = load_assessments()

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = []
    for a in assessments:
        text = f"{a.get('name','')} {a.get('description','')} {' '.join(a.get('keys', []))}"
        texts.append(text)

    print("Creating embeddings...")
    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    print("Embedding ready!")


def get_index():
    return index


def get_model():
    return model


def get_assessments():
    return assessments