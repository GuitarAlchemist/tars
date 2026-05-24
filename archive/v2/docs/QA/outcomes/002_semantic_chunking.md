# Semantic Chunking Strategy Outcome

## Goal

The goal was to improve the quality of data ingestion for Semantic Memory by moving beyond simple fixed-size chunking. Fixed-size chunking often breaks context, splitting sentences and logical blocks (like code) in arbitrary places, which degrades retrieval quality.

## Implementation

We implemented a **Semantic Chunking** strategy in `Tars.Cortex.Chunking`. This strategy:

1. Splits text into sentences.
2. Computes embeddings for each sentence using the LLM service.
3. Groups sentences into chunks based on cosine similarity of their embeddings.
4. Respects `MinChunkSize` and `ChunkSize` constraints.

## Results

We compared **Fixed Size** (100 chars) vs. **Semantic** (200 chars max, 0.7 similarity threshold) on a sample technical text.

### Fixed Size Chunking (Baseline)

Notice how words and logical thoughts are broken:

- `[demo_chunk_0] ... It co`
- `[demo_chunk_1] nsists of two main components...`
- `[demo_chunk_3] ... Collecti`
- `[demo_chunk_4] onName: string...`

### Semantic Chunking (New)

Chunks respect boundaries and semantic meaning:

- `[demo_chunk_0] # Semantic Memory in TARS v2\n\nSemantic memory allows the agent to learn from past experiences.`
- `[demo_chunk_5] Here is a sample configuration:\n\n```fsharp\ntype RagConfig = {\n    CollectionName: string\n    TopK: int\n    MinScore: float32\n}\n```\n\nWhen a new task is executed, the agent queries the semantic memory to find relevant past experiences.`

The semantic chunking successfully kept the code block together with its context and avoided splitting words or sentences.

## Conclusion

Semantic chunking provides significantly better context preservation for RAG workflows. This will be the default strategy for text ingestion in TARS v2.
