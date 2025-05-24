# TARS DSL Grammar Specification

This document provides a formal grammar specification for the TARS Domain Specific Language (DSL).

## Notation

The grammar is specified using a variant of Extended Backus-Naur Form (EBNF):

- `|` denotes alternatives
- `[ ... ]` denotes optional elements
- `{ ... }` denotes elements that can be repeated zero or more times
- `( ... )` denotes grouping
- `" ... "` denotes literal strings
- `' ... '` denotes literal characters
- `/* ... */` denotes comments

## Lexical Elements

### Whitespace

```
whitespace ::= ' ' | '\t' | '\n' | '\r' | '\r\n'
```

### Comments

```
comment ::= single_line_comment | multi_line_comment
single_line_comment ::= "//" { any_character_except_newline }
multi_line_comment ::= "/*" { any_character_except_end_comment } "*/"
```

### Identifiers

```
identifier ::= letter { letter | digit | '_' }
letter ::= 'a' | 'b' | ... | 'z' | 'A' | 'B' | ... | 'Z'
digit ::= '0' | '1' | ... | '9'
```

### Literals

```
literal ::= string_literal | raw_string_literal | number_literal | boolean_literal | null_literal
string_literal ::= '"' { string_character | escape_sequence | interpolation } '"'
raw_string_literal ::= '"""' { any_character_except_triple_quote } '"""'
number_literal ::= [ '-' | '+' ] digit { digit } [ '.' digit { digit } ] [ ('e' | 'E') [ '-' | '+' ] digit { digit } ]
boolean_literal ::= "true" | "false"
null_literal ::= "null"
```

### String Characters

```
string_character ::= any_character_except_quote_backslash_dollar
escape_sequence ::= '\' ( '"' | '\' | 'n' | 'r' | 't' )
interpolation ::= "${" expression "}"
```

### Variable References

```
variable_reference ::= '@' identifier
```

## Syntactic Elements

### Program

```
program ::= { block }
```

### Block

```
block ::= block_type [ identifier ] '{' block_body '}'
block_type ::= "CONFIG" | "PROMPT" | "ACTION" | "TASK" | "AGENT" | "AUTO_IMPROVE" | "DESCRIBE" | "SPAWN_AGENT" | "MESSAGE" | "SELF_IMPROVE" | "TARS" | "COMMUNICATION" | "VARIABLE" | "IF" | "ELSE" | "FOR" | "WHILE" | "FUNCTION" | "CALL" | "RETURN" | "IMPORT" | "INCLUDE" | "EXPORT" | "TEMPLATE" | "USE_TEMPLATE"
block_body ::= property_block | content_block
```

### Property Block

```
property_block ::= { property } { block }
property ::= identifier ':' property_value [ ',' ]
property_value ::= literal | variable_reference | expression | list_literal | object_literal | import_value | template_value
```

### Content Block

```
content_block ::= "```" { any_character_except_triple_backtick } "```" { block }
```

### List Literal

```
list_literal ::= '[' [ property_value { ',' property_value } [ ',' ] ] ']'
```

### Object Literal

```
object_literal ::= '{' [ property { ',' property } [ ',' ] ] '}'
```

### Expression

```
expression ::= term { binary_operator term }
term ::= literal | variable_reference | list_literal | object_literal | '(' expression ')'
binary_operator ::= '+' | '-' | '*' | '/' | '%' | '^'
unary_operator ::= '+' | '-'
```

### Import and Include

```
import_block ::= "IMPORT" '{' string_literal '}'
include_block ::= "INCLUDE" '{' string_literal '}'
import_value ::= "import" string_literal
```

### Template

```
template_block ::= "TEMPLATE" [ identifier ] '{' { property } { block } '}'
use_template_block ::= "USE_TEMPLATE" identifier '{' { property } '}'
template_value ::= "template" string_literal
```

## Examples

### Basic Block

```
CONFIG {
    name: "My Config",
    version: "1.0",
    description: "A sample configuration"
}
```

### Block with Nested Blocks

```
FUNCTION add {
    parameters: "a, b",
    description: "Adds two numbers",
    
    VARIABLE result {
        value: @a + @b,
        description: "The result of adding a and b"
    }
    
    RETURN {
        value: @result
    }
}
```

### Content Block

```
PROMPT {
```
This is a content block.
It can contain multiple lines of text.
It can also contain special characters like { and }.
```
}
```

### String Interpolation

```
VARIABLE message {
    value: "Hello, ${name}!",
    description: "A greeting message"
}
```

### Raw String Literal

```
VARIABLE multiline {
    value: """
This is a raw string literal.
It can contain multiple lines of text.
It can also contain special characters like { and }.
It can even contain quotes like " without escaping.
""",
    description: "A multiline string"
}
```

### Variable References

```
VARIABLE x {
    value: 42,
    description: "The answer"
}

VARIABLE y {
    value: @x,
    description: "A reference to x"
}
```

### Expressions

```
VARIABLE a {
    value: 10,
    description: "First number"
}

VARIABLE b {
    value: 20,
    description: "Second number"
}

VARIABLE sum {
    value: @a + @b,
    description: "The sum of a and b"
}

VARIABLE product {
    value: @a * @b,
    description: "The product of a and b"
}
```

### Imports and Includes

```
IMPORT {
    "path/to/file.tars"
}

INCLUDE {
    "path/to/file.tars"
}
```

### Templates

```
TEMPLATE button {
    type: "button",
    style: "primary",
    text: "Click me",
    action: "submit"
}

USE_TEMPLATE button {
    text: "Submit",
    action: "submit_form"
}
```

## Railroad Diagrams

Railroad diagrams are a visual representation of the grammar rules. They show the possible paths through the grammar, making it easier to understand the syntax.

### Program

```
program ::= { block }
```

```
Program
┌───────────────┐
│               │
└─┬─► Block ─┬──┘
  │          │
  └──────────┘
```

### Block

```
block ::= block_type [ identifier ] '{' block_body '}'
```

```
Block
┌─► Block Type ─┬─────────────┬─► '{' ─► Block Body ─► '}' ─┐
                │             │
                └─► Identifier┘
```

### Property Block

```
property_block ::= { property } { block }
```

```
Property Block
┌───────────────┐  ┌───────────────┐
│               │  │               │
└─┬─► Property ┬┴──┴─┬─► Block ─┬──┘
  │            │     │          │
  └────────────┘     └──────────┘
```

### Content Block

```
content_block ::= "```" { any_character_except_triple_backtick } "```" { block }
```

```
Content Block
┌─► "```" ─┬───────────────────────────────────┬─► "```" ─┬───────────────┬─┐
           │                                   │          │               │
           └─► Any Character Except Triple Backtick ◄─────┘          └─┬─► Block ─┬──┘
                                                                       │          │
                                                                       └──────────┘
```

### Expression

```
expression ::= term { binary_operator term }
```

```
Expression
┌─► Term ─┬───────────────────────────┬─┐
          │                           │
          └─► Binary Operator ─► Term ┘
```

### Term

```
term ::= literal | variable_reference | list_literal | object_literal | '(' expression ')'
```

```
Term
┌─► Literal ──────────────┐
│                         │
├─► Variable Reference ───┤
│                         │
├─► List Literal ─────────┤
│                         │
├─► Object Literal ───────┤
│                         │
└─► '(' ─► Expression ─► ')'
```

## Conclusion

This grammar specification provides a formal definition of the TARS DSL syntax. It can be used as a reference for implementing parsers, syntax highlighters, and other tools that work with the TARS DSL.
