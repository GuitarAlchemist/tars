# TARS Grammars

This directory contains central grammar definitions for the TARS project.

## Files

* **`cortex.ebnf`**: The Extended Backus-Naur Form (EBNF) grammar definition for the Cortex language/DSL.
* **`fsharp.ebnf`**: Simplified F# grammar to guide LLM code generation towards idiomatic styles.
* **`cortex_grammar.md`**: Documentation and explanation of the Cortex grammar.

## Usage

These files serve as the single source of truth for grammar definitions used across different components (e.g., Tars.Cortex, Tars.Core). Changes to the grammar should be made here and propagated to the respective parsers.
