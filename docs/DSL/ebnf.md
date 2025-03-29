# TARS DSL EBNF Specification

This document provides the Extended Backus-Naur Form (EBNF) specification for the TARS Domain Specific Language (DSL).

## EBNF Notation

The EBNF notation used in this document follows these conventions:
- Terminal symbols are enclosed in single quotes: 'if'
- Non-terminal symbols are written without quotes: expression
- Optional elements are enclosed in square brackets: [expression]
- Elements that can appear zero or more times are enclosed in curly braces: {statement}
- Alternatives are separated by vertical bars: 'if' | 'while'
- Grouping is indicated by parentheses: ('if' | 'while')
- The definition symbol is ::=

## EBNF Specification

```ebnf
(* TARS DSL - Extended Backus-Naur Form Specification *)
(* Generated on: 2025-03-29 12:00:00 UTC *)

(* Top-level program structure *)
<tars-program> ::= { <block> }
<block> ::= <block-type> <block-name>? '{' <block-content> '}'
<block-type> ::= 'CONFIG' | 'PROMPT' | 'ACTION' | 'TASK' | 'AGENT' | 'AUTO_IMPROVE' | 'DATA' | 'TOOLING'
<block-name> ::= <identifier>
<block-content> ::= { <property> | <statement> | <block> }

(* Property definitions *)
<property> ::= <identifier> ':' <value> ';'?
<value> ::= <string> | <number> | <boolean> | <array> | <object> | <identifier>
<string> ::= '"' { <any-character-except-double-quote> | '\"' } '"'
<number> ::= <integer> | <float>
<integer> ::= ['-'] <digit> { <digit> }
<float> ::= <integer> '.' <digit> { <digit> }
<boolean> ::= 'true' | 'false'
<array> ::= '[' [ <value> { ',' <value> } ] ']'
<object> ::= '{' [ <property> { ',' <property> } ] '}'

(* Statement definitions *)
<statement> ::= <assignment> | <function-call> | <control-flow> | <return-statement>
<assignment> ::= ['let'] <identifier> ['=' | ':='] <expression> ';'?
<function-call> ::= <identifier> '(' [ <expression> { ',' <expression> } ] ')' ';'?
<control-flow> ::= <if-statement> | <for-loop> | <while-loop>
<if-statement> ::= 'if' '(' <expression> ')' '{' <block-content> '}' [ 'else' '{' <block-content> '}' ]
<for-loop> ::= 'for' '(' <assignment> ';' <expression> ';' <expression> ')' '{' <block-content> '}'
<while-loop> ::= 'while' '(' <expression> ')' '{' <block-content> '}'
<return-statement> ::= 'return' <expression>? ';'?

(* Expression definitions *)
<expression> ::= <value> | <identifier> | <function-call> | <binary-expression> | '(' <expression> ')'
<binary-expression> ::= <expression> <operator> <expression>
<operator> ::= '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '>' | '<=' | '>=' | '&&' | '||'

(* Identifier definition *)
<identifier> ::= <letter> { <letter> | <digit> | '_' }
<letter> ::= 'A' | 'B' | ... | 'Z' | 'a' | 'b' | ... | 'z'
<digit> ::= '0' | '1' | ... | '9'

(* Specific block definitions *)
<config-block> ::= 'CONFIG' '{' { <config-property> } '}'
<config-property> ::= 'version' ':' <string> ';'? | 'author' ':' <string> ';'? | 'description' ':' <string> ';'?

<prompt-block> ::= 'PROMPT' [<identifier>] '{' <prompt-content> '}'
<prompt-content> ::= <string> | { <property> }

<action-block> ::= 'ACTION' [<identifier>] '{' { <statement> } '}'

<task-block> ::= 'TASK' [<identifier>] '{' { <property> | <action-block> } '}'

<agent-block> ::= 'AGENT' [<identifier>] '{' { <property> | <task-block> | <communication-block> } '}'
<communication-block> ::= 'COMMUNICATION' '{' { <property> } '}'

<auto-improve-block> ::= 'AUTO_IMPROVE' [<identifier>] '{' { <property> | <statement> } '}'

<data-block> ::= 'DATA' '{' { <assignment> | <statement> } '}'

<tooling-block> ::= 'TOOLING' '{' { <generate-grammar-block> | <diagnostics-block> | <instrumentation-block> } '}'
<generate-grammar-block> ::= 'GENERATE_GRAMMAR' '{' { <property> } '}'
<diagnostics-block> ::= 'DIAGNOSTICS' '{' { <property> } '}'
<instrumentation-block> ::= 'INSTRUMENTATION' '{' { <property> } '}'
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

- The EBNF specification is a formal definition of the TARS DSL syntax
- It can be used to generate parsers and validators for the language
- The specification is subject to change as the language evolves
- Extensions to the language should follow the patterns established in this specification
