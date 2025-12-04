# Agentic Chunking Strategy Outcome

## Goal

The goal was to implement **Agentic Chunking**, a strategy that uses an LLM to intelligently segment complex text (such as episode traces) into logical sections. This is superior to heuristic-based splitting for unstructured or semi-structured data where "meaning" is defined by high-level concepts (e.g., "Problem Statement", "Plan", "Execution") rather than just paragraphs or sentences.

## Implementation

We implemented `agenticChunkAsync` in `Tars.Cortex.Chunking`. This function:

1. Accepts a text document and a `completer` function (LLM interface).
2. Prompts the LLM to analyze the text and split it into sections using a specific format (`### SECTION: [Title] ... ### END SECTION`).
3. Parses the LLM response to create `Chunk` objects with the identified content and metadata.

## Results

We tested the strategy using a demo command `tars smem demo-chunking`.

### Input Text

A markdown document describing Semantic Memory, containing headers, lists, and a code block.

### Agentic Chunking Output

The LLM successfully identified 3 distinct logical sections:

1. **Introduction**:

    ```text
    # Semantic Memory in TARS v2
    Semantic memory allows the agent to learn from past experiences...
    ```

2. **Implementation**:

    ```text
    ## Implementation Details
    The system uses vector embeddings...
    [Code Block]
    ```

3. **Process & Goal**:

    ```text
    When a new task is executed...
    The goal is to improve performance...
    ```

### Comparison

- **Fixed Size**: Arbitrarily split the text, breaking words and code blocks.
- **Semantic**: Grouped sentences by similarity, which worked well but produced more granular chunks (10 chunks).
- **Agentic**: Produced high-level, coherent sections (3 chunks) that perfectly matched the document's logical structure.

## Conclusion

Agentic Chunking is highly effective for summarizing and segmenting complex documents where high-level structure is more important than granular retrieval. It will be the primary strategy for processing **Episode Traces** in the Semantic Memory "Grow" cycle.
