# Official F# Grammar References

Since the F# compiler repository structure evolves, we reference the primary sources for the grammar:

## F# Language Specification

The normative reference for F# grammar is the **F# Language Specification**.

- **Source**: [fsharp.org/specs/](https://fsharp.org/specs/)
- **Appendix A**: Contains the full grammar summary.

## F# Compiler Parser (`pars.fsy`)

The actual implementation of the parser used by the compiler is written in `fsyacc` format.

- **Repository**: [dotnet/fsharp](https://github.com/dotnet/fsharp)
- **Path**: Typically located at `src/Compiler/pars.fsy` or `src/Compiler/Parsing/pars.fsy`.
- **Search**: [Search for `pars.fsy` in dotnet/fsharp](https://github.com/dotnet/fsharp/search?q=pars.fsy)

## Usage in TARS

For LLM code generation, we use the simplified `fsharp.ebnf` in this directory, as the full compiler grammar is too verbose and contains implementation details (like error recovery) not needed for generative tasks.
