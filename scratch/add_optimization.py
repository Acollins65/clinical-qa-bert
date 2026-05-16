import json
import os

notebook_path = r"c:\Users\MSC1\OneDrive - Liverpool John Moores University\Desktop\School files\Advanced deep learning\cw2\cw2-main\Evaluation_and_Deployment.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "optimization-header",
        "metadata": {},
        "source": [
            "## Pipeline Optimization: Granular Chunking vs. SQuAD Paragraphs\n",
            "\n",
            "To further optimize the RAG pipeline, we evaluate a more granular retrieval strategy using the `chunks.json` corpus (384-token windows with 50-token overlap). Unlike standard SQuAD paragraphs, which vary wildly in length, these fixed-size windows ensure a consistent context for the dense retriever and the reader.\n",
            "\n",
            "We also switch to the `multi-qa-mpnet-base-dot-v1` encoder, which is specifically fine-tuned for semantic search across diverse QA datasets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "optimized-retrieval",
        "metadata": {},
        "outputs": [],
        "source": [
            "NEW_FAISS_MODEL = \"sentence_transformers/multi-qa-mpnet-base-dot-v1\"\n",
            "print(f\"Switching to optimized retrieval model: {NEW_FAISS_MODEL}\")\n",
            "opt_encoder = SentenceTransformer(NEW_FAISS_MODEL)\n",
            "\n",
            "with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:\n",
            "    knowledge_chunks = json.load(f)\n",
            "chunk_texts = [c['text'] for c in knowledge_chunks]\n",
            "\n",
            "print(f\"Encoding {len(chunk_texts):,} granular chunks...\")\n",
            "chunk_embeddings = opt_encoder.encode(chunk_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n",
            "\n",
            "opt_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])\n",
            "opt_index.add(chunk_embeddings)\n",
            "\n",
            "print(\"\\nCalculating Optimized Recall@k...\")\n",
            "opt_hits_at = {1: 0, 5: 0, 10: 0}\n",
            "q_embs = opt_encoder.encode(questions_text, show_progress_bar=False, normalize_embeddings=True)\n",
            "\n",
            "for q_emb, gold_ans_list in zip(q_embs, gold_texts):\n",
            "    _, indices = opt_index.search(q_emb.reshape(1, -1), 10)\n",
            "    retrieved_chunks = [chunk_texts[idx] for idx in indices[0]]\n",
            "    for k in (1, 5, 10):\n",
            "        if any(any(a.lower() in c.lower() for a in gold_ans_list) for c in retrieved_chunks[:k]):\n",
            "            opt_hits_at[k] += 1\n",
            "\n",
            "print(f\"\\nOptimized Recall Results (n={len(questions_text)})\")\n",
            "for k in (1, 5, 10):\n",
            "    print(f\"  Recall@{k:<2} : {opt_hits_at[k] / len(questions_text) * 100:.2f}%\")"
        ]
    }
]

# Insert before the demo table
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == "demo-header":
        nb['cells'] = nb['cells'][:i] + new_cells + nb['cells'][i:]
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Optimization section added successfully.")
