# TARS DSL BNF Specification

This document provides the Backus-Naur Form (BNF) specification for the TARS Domain Specific Language (DSL).

## BNF Notation

The BNF notation used in this document follows these conventions:
- Non-terminal symbols are enclosed in angle brackets: `<expression>`
- Terminal symbols are enclosed in double quotes: `"if"`
- The definition symbol is `::=`
- Alternatives are separated by vertical bars: `|`
- Optional elements are not directly represented in BNF (they are represented as alternatives)
- Elements that can appear zero or more times are not directly represented in BNF (they are represented using recursion)

## BNF Specification

```bnf
# TARS DSL - Backus-Naur Form Specification
# Generated on: 2025-03-29 12:00:00 UTC

# Top-level program structure
<tars-program> ::= <block> | <tars-program> <block>
<block> ::= <block-type> <block-name> "{" <block-content> "}" | <block-type> "{" <block-content> "}"
<block-type> ::= "CONFIG" | "PROMPT" | "ACTION" | "TASK" | "AGENT" | "AUTO_IMPROVE" | "DATA" | "TOOLING"
<block-name> ::= <identifier>
<block-content> ::= <property> | <statement> | <block> | <block-content> <property> | <block-content> <statement> | <block-content> <block>

# Property definitions
<property> ::= <identifier> ":" <value> ";" | <identifier> ":" <value>
<value> ::= <string> | <number> | <boolean> | <array> | <object> | <identifier>
<string> ::= "\"" <string-content> "\""
<string-content> ::= <empty> | <character> | <string-content> <character>
<number> ::= <integer> | <float>
<integer> ::= <digit> | <integer> <digit>
<float> ::= <integer> "." <integer>
<boolean> ::= "true" | "false"
<array> ::= "[" <array-content> "]" | "[" "]"
<array-content> ::= <value> | <array-content> "," <value>
<object> ::= "{" <object-content> "}" | "{" "}"
<object-content> ::= <property> | <object-content> "," <property>

# Statement definitions
<statement> ::= <assignment> | <function-call> | <control-flow> | <return-statement>
<assignment> ::= "let" <identifier> "=" <expression> ";" | <identifier> "=" <expression> ";" | "let" <identifier> "=" <expression> | <identifier> "=" <expression>
<function-call> ::= <identifier> "(" <argument-list> ")" ";" | <identifier> "(" ")" ";" | <identifier> "(" <argument-list> ")" | <identifier> "(" ")"
<argument-list> ::= <expression> | <argument-list> "," <expression>
<control-flow> ::= <if-statement> | <for-loop> | <while-loop>
<if-statement> ::= "if" "(" <expression> ")" "{" <block-content> "}" | "if" "(" <expression> ")" "{" <block-content> "}" "else" "{" <block-content> "}"
<for-loop> ::= "for" "(" <assignment> ";" <expression> ";" <expression> ")" "{" <block-content> "}"
<while-loop> ::= "while" "(" <expression> ")" "{" <block-content> "}"
<return-statement> ::= "return" <expression> ";" | "return" ";" | "return" <expression> | "return"

# Expression definitions
<expression> ::= <value> | <identifier> | <function-call> | <binary-expression> | "(" <expression> ")"
<binary-expression> ::= <expression> <operator> <expression>
<operator> ::= "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||"

# Identifier definition
<identifier> ::= <letter> | <identifier> <letter> | <identifier> <digit> | <identifier> "_"
<letter> ::= "A" | "B" | ... | "Z" | "a" | "b" | ... | "z"
<digit> ::= "0" | "1" | ... | "9"
<empty> ::= 
<character> ::= <letter> | <digit> | <special-character>
<special-character> ::= " " | "!" | "#" | "$" | ... | "{" | "|" | "}" | "~"
```

## Examples

### Example 1: Simple Configuration

```
CONFIG {
    version: "1.0";
    author: "John Doe";
    description: "Example TARS program";
}
```

### Example 2: Prompt with Properties

```
PROMPT {
    text: "Generate a list of 5 ideas for improving code quality.";
    model: "gpt-4";
    temperature: 0.7;
}
```

### Example 3: Action with Statements

```
ACTION {
    let result = processFile("example.cs");
    print(result);
    
    if (result.issues.length > 0) {
        for (let i = 0; i < result.issues.length; i++) {
            print("Issue " + (i + 1) + ": " + result.issues[i]);
        }
    } else {
        print("No issues found!");
    }
}
```

### Example 4: Agent with Tasks

```
AGENT CodeAnalyzer {
    capabilities: ["code-analysis", "refactoring"];
    
    TASK AnalyzeCode {
        description: "Analyze code quality";
        
        ACTION {
            let code = readFile("example.cs");
            return analyzeCode(code);
        }
    }
    
    TASK RefactorCode {
        description: "Refactor code based on analysis";
        dependencies: [AnalyzeCode];
        
        ACTION {
            let analysis = getTaskResult(AnalyzeCode);
            let code = readFile("example.cs");
            let refactored = refactorCode(code, analysis);
            writeFile("example_refactored.cs", refactored);
        }
    }
}
```

## Notes

- The BNF specification is a formal definition of the TARS DSL syntax
- It is more restrictive than the EBNF specification, as it does not directly support optional elements or repetition
- It can be used to generate parsers and validators for the language
- The specification is subject to change as the language evolves
- Extensions to the language should follow the patterns established in this specification
