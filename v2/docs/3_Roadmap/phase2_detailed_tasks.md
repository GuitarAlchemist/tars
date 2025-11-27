# Phase 2 Detailed Tasks: The Brain & Cognitive Hardening

## 2.2 Memory Grid (Simple Persistence)

- [x] **2.2.1 Persistence Layer**
  - [x] Update `InMemoryVectorStore` to include `PersistToFileAsync(path: string)`
  - [x] Update `InMemoryVectorStore` to include `LoadFromFileAsync(path: string)`
  - [x] Ensure atomic writes (write to temp, then move) to prevent corruption.
- [x] **2.2.2 Metadata Enhancements**
  - [x] Update `VectorEntry` record with:
    - `Version: int`
    - `CreatedAt: DateTime`
    - `LastUsed: DateTime`
  - [x] Update `SaveAsync` to increment version if ID exists.
  - [x] Update `SearchAsync` to update `LastUsed` (maybe async background task?).

## 2.4 Internal Knowledge Graph (Graphiti-style)

- [x] **2.4.1 Graph Domain**
  - [x] Create `src/Tars.Graph/Domain.fs`
- [ ] **2.5.2 Units of Measure**
  - [ ] Define `[<Measure>] type token`
  - [ ] Define `[<Measure>] type ms`
  - [ ] Update `BudgetGovernor` to use these.
