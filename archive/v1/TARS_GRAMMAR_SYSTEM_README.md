# 🧬 TARS Extended DSL Support: Grammar & Multi-Language System

## 🎯 Overview

This implementation extends TARS with comprehensive support for:

- **Multi-language code blocks** with `LANG("LANGUAGE")` syntax
- **EBNF grammar management** (inline and external)
- **RFC-based standards integration**
- **Hybrid grammar storage** using FileInfo-based resolution

## 🏗️ Architecture

### Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `GrammarSource.fs` | Core types and grammar handling | `Tars.Engine.Grammar/` |
| `GrammarResolver.fs` | Grammar resolution and management | `Tars.Engine.Grammar/` |
| `LanguageDispatcher.fs` | Multi-language code execution | `Tars.Engine.Grammar/` |
| `RFCProcessor.fs` | RFC parsing and grammar extraction | `Tars.Engine.Grammar/` |

### Directory Structure

```
.tars/
  grammars/
    MiniQuery.tars              # Example query language grammar
    RFC3986_URI.tars            # URI syntax from RFC 3986
    grammar_index.json          # Grammar metadata index

Tars.Engine.Grammar/
  GrammarSource.fs              # Core types and utilities
  GrammarResolver.fs            # Resolution and file management
  LanguageDispatcher.fs         # Language execution engine
  RFCProcessor.fs               # RFC processing and extraction
  Tars.Engine.Grammar.fsproj    # Project file

tools/
  tarscli_grammar.fsx           # CLI tool for grammar management

examples/
  agent_meta_script.tars        # Multi-language agent example
  agent_with_inline.tars        # Inline grammar example
```

## 🚀 Features

### 1. Multi-Language Support

Use `LANG("LANGUAGE")` blocks for polyglot development:

```tars
logic {
  LANG("FSHARP") {
    let processData data =
        data |> List.filter (fun x -> x > 0)
  }
  
  LANG("PYTHON") {
    def analyze_data(data):
        return {"mean": sum(data) / len(data), "count": len(data)}
  }
  
  LANG("CSHARP") {
    public class DataProcessor {
        public static string FormatResults(object data) {
            return JsonSerializer.Serialize(data);
        }
    }
  }
}
```

**Supported Languages:**
- F# (`FSHARP`)
- C# (`CSHARP`) 
- Python (`PYTHON`)
- Rust (`RUST`)
- JavaScript (`JAVASCRIPT`)
- TypeScript (`TYPESCRIPT`)
- PowerShell (`POWERSHELL`)
- Bash (`BASH`)
- SQL (`SQL`)

### 2. Grammar Management

#### External Grammars (Recommended for Reuse)

```tars
// Reference external grammar
use_grammar("MiniQuery")
use_grammar("RFC3986_URI")
```

#### Inline Grammars (For Experimentation)

```tars
grammar "MathExpression" {
  LANG("EBNF") {
    expression = term , { ( "+" | "-" ) , term } ;
    term = factor , { ( "*" | "/" ) , factor } ;
    factor = number | "(" , expression , ")" ;
    number = digit , { digit } ;
    digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
  }
}
```

### 3. RFC Integration

```tars
rfc "rfc3986" {
  title: "URI Generic Syntax"
  url: "https://datatracker.ietf.org/doc/html/rfc3986"
  extract_rules: ["URI", "scheme", "authority", "path"]
  verify_compatibility_with: "UriValidatorAgent"
  use_in: "test_suite"
}
```

## 🛠️ CLI Tools

### Grammar Management Commands

```bash
# List all available grammars
dotnet fsi tools/tarscli_grammar.fsx list

# Extract inline grammars to external files
dotnet fsi tools/tarscli_grammar.fsx extract-from-meta agent.tars

# Show grammar content for inlining
dotnet fsi tools/tarscli_grammar.fsx inline MiniQuery

# Compute grammar hash for versioning
dotnet fsi tools/tarscli_grammar.fsx hash RFC3986_URI

# Generate grammar from examples (placeholder)
dotnet fsi tools/tarscli_grammar.fsx generate-from-examples examples.txt

# Download and process RFC grammar
dotnet fsi tools/tarscli_grammar.fsx download-rfc rfc3986
```

### Example Output

```
📚 Available TARS Grammars:
==========================
📄 MiniQuery (manual) - vv1.0 - 2025-06-07
   📁 .tars/grammars\MiniQuery.tars (995 bytes)

📄 RFC3986_URI (rfc) - vv1.0 - 2025-06-07
   📁 .tars/grammars\RFC3986_URI.tars (3580 bytes)
```

## 📝 Grammar File Format

Each `.tars` grammar file includes metadata and EBNF definition:

```tars
meta {
  name: "MiniQuery"
  version: "v1.0"
  source: "manual"
  language: "EBNF"
  created: "2025-06-07 10:21:00"
  description: "Simple query language for file operations"
  tags: ["query", "dsl", "files"]
}

grammar {
  LANG("EBNF") {
    query = "find" , ws , identifier , ws , "in" , ws , target ;
    target = identifier | string_literal ;
    // ... more rules
  }
}
```

## 🧪 Testing

### Language Block Validation

The system validates:
- ✅ Language runtime availability
- ✅ Code syntax (basic checks)
- ✅ Dependencies and entry points

### Grammar Validation

The system checks:
- ✅ EBNF syntax correctness
- ✅ Balanced braces and brackets
- ✅ Rule completeness

## 🔄 Resolution Hierarchy

Grammar resolution follows this order:

1. **External file**: `.tars/grammars/{name}.tars`
2. **Inline definition**: Current `.tars` file
3. **Error**: Grammar not found

## 📊 Benefits

### For TARS Development

- **Polyglot Agents**: Use the best language for each task
- **Standards Compliance**: Validate against RFCs automatically
- **Grammar Evolution**: Version and track grammar changes
- **Code Generation**: Auto-generate parsers from grammars

### For Agent Behavior

- **Language-Aware Reasoning**: Agents can choose optimal languages
- **Cross-Language Refactoring**: Transform code between languages
- **Spec-Based Validation**: Ensure compliance with standards
- **Grammar-Driven Parsing**: Generate parsers on-the-fly

## 🚀 Next Steps

### Immediate Enhancements

1. **Parser Generation**: Auto-generate F# parsers from EBNF
2. **Language Interop**: Enable cross-language data passing
3. **RFC Automation**: Auto-download and process well-known RFCs
4. **Grammar Visualization**: Generate railroad diagrams

### Advanced Features

1. **ML Grammar Generation**: Infer grammars from examples
2. **Grammar Optimization**: Simplify and optimize rules
3. **Cross-Grammar Analysis**: Find similarities between grammars
4. **Agent Grammar Evolution**: Let agents evolve their own grammars

## 🎉 Success Metrics

- ✅ **Multi-language support** implemented with 9 languages
- ✅ **Hybrid grammar storage** with FileInfo-based resolution
- ✅ **CLI tooling** for grammar management
- ✅ **RFC integration** framework established
- ✅ **Example grammars** (MiniQuery, RFC3986_URI) created
- ✅ **Validation systems** for both grammars and language blocks

**TARS now has a comprehensive grammar brain that supports polyglot development, standards-based validation, and evolutionary grammar management!**
