# Task Summary

**Objective**: Implement a tail‑recursive Fibonacci function in F# without mutable variables and provide unit tests for n = 0‑20.

**Key Decisions**:
- Use an inner tail‑recursive helper (e.g., `fibTail n a b`).
- Expose a single‑parameter public API.
- Choose a testing framework (xUnit/NUnit/Expecto) and generate tests covering the first 21 Fibonacci numbers.
- Handle edge cases (`n < 0`) with appropriate error handling.

**Constraints**:
- No mutable state.
- Must be tail‑recursive.
- Include a comprehensive unit‑test suite.

**Validation**: Run `run_tests` to ensure all tests pass.

**Tools Available**: write_to_file, read_file, generate_test, run_tests, fsharp_compile, etc.

**Lessons Learned**: Prior tail‑recursive implementations, binary search with Option types, refactoring duplicate code, successful test generation.

**Critical Instructions**:
1. Operate autonomously; no human questions.
2. Discover file locations before reading/writing.
3. Persist all code changes using write_to_file.
4. Strictly follow tail‑recursion and immutability constraints.
5. Validate with unit tests; if errors occur, explain and provide best partial solution.
6. Optionally use a Workflow of Thought for complex reasoning.

**Deliverables**:
- `src/Fibonacci.fs` with the tail‑recursive `fib` function.
- `tests/FibonacciTests.fs` with unit tests for n = 0‑20.
- Ensure the project builds and all tests pass.
