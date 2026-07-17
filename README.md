# 🔍 DIC: Decompose-Index-Compose Fashion Search Engine

> An AI-powered multimodal fashion image search engine that goes beyond vanilla CLIP to solve compositionality in fashion retrieval.

## 🏗️ Architecture: DIC + VLM-as-Judge

This system uses a **4-stage cascade pipeline** that decomposes images into structured attributes, indexes multiple representations, and composes matches at query time with compositional verification.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXER (Offline)                            │
│                                                                     │
│  Image → BLIP Caption → LLM Attribute Extraction → Structured JSON │
│  Image → CLIP Embedding (visual)                                    │
│  Caption → SentenceTransformer Embedding (textual)                  │
│  All stored in ChromaDB with rich metadata                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       RETRIEVER (Online)                            │
│                                                                     │
│  Stage 1: Metadata Pre-Filter (color, environment, style)           │
│  Stage 2: Dual-Channel Vector Search (CLIP + Text embeddings)       │
│  Stage 3: Compositional Re-Ranking (garment-level pairwise match)   │
│  Stage 4: VLM-as-Judge (vision model scores final candidates)       │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Not Vanilla CLIP?

CLIP encodes an entire image/text into a single vector, which collapses attribute-object bindings. For example, "red shirt with blue pants" and "blue shirt with red pants" produce nearly identical CLIP embeddings. Our DIC architecture solves this through:

1. **Structured attribute extraction** — explicitly binds colors to garments
2. **Metadata pre-filtering** — eliminates non-matching images before vector search
3. **Compositional re-ranking** — pairwise verification that "red" is on the "shirt", not the "pants"
4. **VLM-as-Judge** — a vision-language model directly looks at the image and confirms the match

## 📁 Project Structure

```
├── config.py                          # Central configuration (all settings in one place)
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
│
├── dataset/
│   ├── download_dataset.py            # Downloads 1000 Fashionpedia images
│   └── images/                        # Downloaded images (gitignored)
│
├── indexer/                           # Part A: The Indexer
│   ├── caption_generator.py           # BLIP image captioning
│   ├── attribute_extractor.py         # LLM structured attribute extraction
│   ├── embedding_generator.py         # CLIP + SentenceTransformer embeddings
│   ├── vector_store.py                # ChromaDB storage wrapper
│   └── build_index.py                 # Main indexing orchestrator
│
├── retriever/                         # Part B: The Retriever
│   ├── query_parser.py                # Parse NL query → structured attributes
│   ├── vector_search.py               # Stage 1+2: Filter + dual vector search
│   ├── compositional_reranker.py      # Stage 3: Garment-level re-ranking
│   ├── vlm_judge.py                   # Stage 4: VLM vision scoring
│   └── search.py                      # Main search orchestrator
│
├── utils/
│   └── visualize.py                   # Result visualization
│
├── vector_store/                      # ChromaDB persistence (gitignored)
└── results/                           # Saved result images for report
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier works)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/glance-fashion-search.git
cd glance-fashion-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 3. Download Dataset

```bash
python dataset/download_dataset.py
```

This downloads 1000 images from Fashionpedia and saves them to `dataset/images/`.

### 4. Build the Index

```bash
python indexer/build_index.py
```

This runs the full indexing pipeline:
- BLIP captioning (~3 sec/image on GPU, ~10 sec/image on CPU)
- LLM attribute extraction via API (~70 min due to rate limits)
- CLIP + SentenceTransformer embeddings
- Storage in ChromaDB

> **💡 Tip:** For faster indexing, run this on [Google Colab](https://colab.research.google.com) with a free GPU. See [Colab Instructions](#-running-on-google-colab) below.

### 5. Search!

```bash
# Full pipeline with VLM judge
python retriever/search.py --query "A red tie and a white shirt in a formal setting"

# Fast mode (skip VLM judge)
python retriever/search.py --query "Casual weekend outfit for a city walk" --no-vlm

# Save result visualization
python retriever/search.py --query "A person in a bright yellow raincoat" --save
```

## ☁️ Running on Google Colab

For faster indexing with a free GPU:

1. Upload the project to Colab or clone from GitHub
2. Install dependencies: `!pip install -r requirements.txt`
3. Create `.env` file with your API key
4. Run: `!python dataset/download_dataset.py`
5. Run: `!python indexer/build_index.py`
6. Download the `vector_store/` folder to your local machine
7. Run queries locally: `python retriever/search.py --query "..."`

**Estimated Colab time:** ~2 hours (within the 4-hour free tier limit)

## 📊 Evaluation Queries

The system is designed to handle these query types:

| Query Type | Example |
|---|---|
| Attribute Specific | "A person in a bright yellow raincoat" |
| Contextual/Place | "Professional business attire inside a modern office" |
| Complex Semantic | "Someone wearing a blue shirt sitting on a park bench" |
| Style Inference | "Casual weekend outfit for a city walk" |
| Compositional | "A red tie and a white shirt in a formal setting" |

## 🧠 Technical Details

### Models Used

| Component | Model | Purpose |
|---|---|---|
| Image Captioning | `Salesforce/blip-image-captioning-large` | Generate detailed image descriptions |
| Attribute Extraction | `meta-llama/llama-4-maverick` (via OpenRouter) | Parse captions → structured JSON |
| Visual Embeddings | `ViT-B/32` (OpenAI CLIP via open_clip) | Global visual similarity |
| Text Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) | Caption semantic similarity |
| VLM Judge | `meta-llama/llama-4-scout` (via OpenRouter) | Final visual verification with reasoning |
| Vector Database | ChromaDB (PersistentClient) | Embedding storage + metadata filtering |

### Retrieval Pipeline

1. **Stage 1 — Metadata Pre-Filter:** ChromaDB `where` clause filters by color, environment, style. Reduces candidates from 1000 → ~80.
2. **Stage 2 — Dual Vector Search:** Fuses CLIP (visual) and SentenceTransformer (textual) similarity scores. 80 → top 20.
3. **Stage 3 — Compositional Re-Rank:** Pairwise garment-level matching verifies attribute-object bindings. 20 → top 10.
4. **Stage 4 — VLM-as-Judge:** Vision model directly scores and explains each match. 10 → top 5 with reasoning.

### Scalability

The cascade architecture scales to 1M+ images:
- Metadata filtering: O(1) with database indexing
- HNSW vector search: O(log N)
- Re-ranking stages: O(k) where k is fixed (20, 10)

## 📄 License

This project was built as part of the Glance ML Internship Assignment.
