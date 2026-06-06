# Outcome: AST Chunking Implementation

## Goal

Implement an AST-based chunking strategy for F# code to improve the quality of code embeddings and retrieval by respecting the syntactic structure of the language (modules, types, functions).

## Implementation Details

- **Strategy**: `Ast` (added to `ChunkingStrategy` enum).
- **Parser**: Uses `FSharp.Compiler.Service` (FCS) to parse F# source code into an Abstract Syntax Tree (AST).
- **Logic**:
  - Parses the code using `FSharpChecker`.
  - Traverses the AST to identify top-level declarations:
    - `SynModuleDecl.Let` (Functions and values)
    - `SynModuleDecl.Types` (Type definitions)
    - `SynModuleDecl.Exception` (Exception definitions)
    - `SynModuleDecl.NestedModule` (Recursive traversal)
  - Extracts the source code for each declaration using its `Range`.
  - Creates a `Chunk` for each declaration.
- **Fallback**: If parsing fails or no chunks are found (e.g., non-F# text), it falls back to `Recursive` chunking.
- **Async**: The implementation `astChunkAsync` is fully asynchronous to handle FCS operations.

## Challenges & Solutions

1. **FCS Version Compatibility**: The `SynModuleDecl.NestedModule` union case has different signatures in different FCS versions (specifically regarding the `trivia` argument).
    - *Solution*: Used positional pattern matching with 6 arguments to match the version used in the project, ensuring compatibility.
2. **Async/Task Mismatch**: FCS `ParseFile` returns `Async<'T>`, while the TARS v2 codebase uses `Task<'T>`.
    - *Solution*: Used `Async.StartAsTask` to bridge the gap.
3. **State Machine Limitations**: The `task { ... }` builder in F# has limitations regarding `let rec` definitions inside the block.
    - *Solution*: Refactored the recursive AST traversal function (`walkDecls`) to be a local function defined *outside* the `task` block, passing necessary context (chunks list) as arguments.

## Verification

- **Build**: Successfully built `Tars.Cortex` and `Tars.Interface.Cli`.
- **Demo**: Ran `tars smem demo-chunking` which now includes an AST Chunking section.
- **Output**: Confirmed that F# code is correctly split into `type`, `let`, and `module` chunks, respecting boundaries that other strategies (like Fixed Size) would ignore.

### Sample Output

```text
[demo_ast_chunk_0] type DemoConfig = {
    Id: string
    Value: int
}
[demo_ast_chunk_1] let processConfig (config: DemoConfig) =
    printfn "Processing %s" config.Id
    config.Value * 2
[demo_ast_chunk_2] type Processor() =
    member this.Run() =
        let config = { Id = "test"; Value = 10 }
        processConfig config
```

## Next Steps

- Integrate AST chunking into the main ingestion pipeline (`CodeGraphIngestor`).
- Add support for other languages (C#, Python) using Roslyn or other parsers if needed (currently F# only).
