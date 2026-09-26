import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# 1. OUR KNOWLEDGE BASE
# ============================================================

documents = [
    """
    RAG stands for Retrieval-Augmented Generation.
    It allows a language model to use external information
    when answering a question.
    """,

    """
    A RAG system normally has two phases: indexing and inference.
    During indexing, documents are collected, split into chunks,
    converted into embeddings, and stored.
    """,

    """
    Embeddings are numerical representations of text.
    Texts with similar meanings have embeddings that are close
    together in vector space.
    """,

    """
    During retrieval, the user's question is converted into
    an embedding. The system compares this embedding with the
    stored document embeddings and retrieves the most similar chunks.
    """,

    """
    After retrieval, the relevant chunks are inserted into the
    prompt as context. The language model then generates an answer
    based on this retrieved information.
    """,

    """
    RAG can reduce hallucinations because the model receives
    relevant external information before generating its answer.
    It can also provide access to private or recently updated data.
    """,
    """Employees receive 25 days of annual leave.""",
    """The capital of the United States is Washington D.C."""
 
]


# ============================================================
# 2. CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating document embeddings...")

document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)

print(f"Created {len(document_embeddings)} embeddings.")
print(f"Embedding dimensions: {len(document_embeddings[0])}")


# ============================================================
# 3. RETRIEVAL
# ============================================================

def retrieve(question, top_k=3):

    # Convert the question into an embedding
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Because the vectors are normalized,
    # dot product = cosine similarity
    similarities = document_embeddings @ question_embedding

    # Sort from highest similarity to lowest
    best_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in best_indices:
        results.append({
            "text": documents[index].strip(),
            "score": float(similarities[index])
        })

    return results


# ============================================================
# 4. GENERATION
# ============================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


def generate_answer(question, retrieved_documents):

    context = "\n\n".join(
        doc["text"] for doc in retrieved_documents
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    return response.output_text


# ============================================================
# 5. COMPLETE RAG PIPELINE
# ============================================================

def rag(question):

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    # RETRIEVE
    retrieved_documents = retrieve(question)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, doc in enumerate(retrieved_documents, start=1):
        print(f"\n#{i}")
        print(f"Similarity: {doc['score']:.3f}")
        print(doc["text"])

    # AUGMENT + GENERATE
    answer = generate_answer(
        question,
        retrieved_documents
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)


# ============================================================
# 6. ASK QUESTIONS
# ============================================================

while True:

    question = input("\nAsk a question (or type 'quit'): ")

    if question.lower() == "quit":
        break

    rag(question)
