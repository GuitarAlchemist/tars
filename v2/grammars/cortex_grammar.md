# Cortex Grammar

This document defines the grammar for the Cortex language.
The formal grammar is defined in [cortex.ebnf](cortex.ebnf).

## EBNF

```ebnf
program ::= goal*
goal    ::= "goal" string_literal "{" task* "}"
task    ::= "task" string_literal
```

## Token Definitions

- `goal`: The literal keyword "goal" (case-insensitive).
- `task`: The literal keyword "task" (case-insensitive).
- `string_literal`: A sequence of characters enclosed in double quotes (`"`).
- `{`, `}`: Curly braces for block delimitation.

## Example

```
goal "MyGoal" {
    task "Task1"
    task "Task2"
}
```
