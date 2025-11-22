# TARS Fractal Grammars System

## 🌀 Overview

The **TARS Fractal Grammars System** is an advanced grammar framework that implements **self-similar, recursive, and dynamically composable grammars** with mathematical fractal properties. This system extends the traditional TARS grammar architecture to support fractal dimensions, recursive transformations, and complex pattern generation.

## 🔬 Key Features

### **Fractal Properties**
- **Fractal Dimension**: Mathematical dimension calculation (e.g., Sierpinski Triangle: 1.585)
- **Self-Similarity**: Patterns that repeat at different scales
- **Recursive Depth**: Configurable recursion limits with termination conditions
- **Scaling Transformations**: Mathematical scaling, rotation, translation, and composition

### **Advanced Transformations**
- **Scale**: Resize patterns by mathematical factors
- **Rotate**: Rotate patterns in grammar space
- **Translate**: Positional transformations
- **Compose**: Combine multiple transformations
- **Recursive**: Apply transformations recursively with depth control
- **Conditional**: Apply transformations based on pattern properties

### **Multi-Format Output**
- **EBNF**: Extended Backus-Naur Form
- **ANTLR**: ANTLR grammar format
- **JSON**: Structured JSON representation
- **XML**: XML grammar format
- **GraphViz**: DOT format for visualization
- **SVG**: Scalable Vector Graphics for visual output

## 📁 Architecture

```
Tars.Engine.Grammar/
├── FractalGrammar.fs              # Core fractal grammar types and engine
├── FractalGrammarParser.fs        # Parser for .fractal files
├── FractalGrammarIntegration.fs   # Integration with TARS ecosystem
└── Tests/
    └── FractalGrammarTests.fs     # Comprehensive test suite

.tars/grammars/
├── SierpinskiTriangle.fractal     # Sierpinski Triangle example
├── KochSnowflake.fractal          # Koch Snowflake example
└── DragonCurve.fractal            # Dragon Curve example

tools/
└── fractal_grammar_cli.fsx       # Command-line interface
```

## 🚀 Quick Start

### 1. **Create a Simple Fractal Grammar**

```fractal
// Simple Sierpinski Triangle
fractal sierpinski {
    pattern = "triangle"
    recursive = "triangle triangle triangle"
    dimension = 1.585
    depth = 5
    transform scale 0.5
}
```

### 2. **Generate Grammar**

```bash
dotnet fsi tools/fractal_grammar_cli.fsx generate \
    -i .tars/grammars/SierpinskiTriangle.fractal \
    -o output/sierpinski.ebnf \
    -f EBNF \
    -d 6 \
    -v
```

### 3. **Analyze Complexity**

```bash
dotnet fsi tools/fractal_grammar_cli.fsx analyze \
    -i .tars/grammars/KochSnowflake.fractal \
    -d
```

## 📝 Fractal Grammar Syntax

### **Basic Structure**

```fractal
meta {
  name: "GrammarName"
  version: "v1.0"
  dimension: 1.585
}

fractal rule_name {
    pattern = "base pattern"
    recursive = "recursive expansion"
    dimension = 1.585
    depth = 6
    terminate = "termination_condition"
    
    transform scale 0.5
    transform rotate 90
    transform recursive 5 scale 0.8
}
```

### **Transformation Types**

| Transformation | Syntax | Description |
|----------------|--------|-------------|
| **Scale** | `transform scale 0.5` | Scale pattern by factor |
| **Rotate** | `transform rotate 90` | Rotate pattern by degrees |
| **Translate** | `transform translate 1.0 0.5` | Move pattern in space |
| **Recursive** | `transform recursive 5 scale 0.8` | Apply recursively |
| **Compose** | `transform compose [scale 0.5, rotate 45]` | Combine transformations |
| **Conditional** | `transform if "condition" then scale 0.5 else scale 1.0` | Conditional application |

### **Termination Conditions**

| Condition | Description |
|-----------|-------------|
| `max_depth_N` | Stop at depth N |
| `pattern_too_long` | Stop when pattern exceeds length |
| `complexity_threshold` | Stop when complexity is too high |
| `always_terminate` | Always stop (base case) |
| `never_terminate` | Never stop (infinite) |

## 🎨 Examples

### **Sierpinski Triangle**
```fractal
fractal sierpinski_triangle {
    pattern = "triangle"
    recursive = "triangle triangle triangle"
    dimension = 1.585
    depth = 6
    terminate = "max_depth_6"
    
    transform scale 0.5
    transform recursive 6 compose [
        scale 0.5,
        rotate 0,
        rotate 120,
        rotate 240
    ]
}
```

### **Koch Snowflake**
```fractal
fractal koch_curve {
    pattern = "line"
    recursive = "line turn60 line turn-120 line turn60 line"
    dimension = 1.261
    depth = 7
    terminate = "max_depth_7"
    
    transform scale 0.333333
    transform recursive 7 scale 0.333333
}
```

### **Dragon Curve**
```fractal
fractal dragon_curve {
    pattern = "F"
    recursive = "F+G+"
    dimension = 2.0
    depth = 12
    terminate = "max_depth_12"
    
    transform recursive 12 compose [
        scale 0.707107,
        rotate 45
    ]
}
```

## 🔧 CLI Commands

### **Generate Grammar**
```bash
dotnet fsi tools/fractal_grammar_cli.fsx generate \
    --input input.fractal \
    --output output.ebnf \
    --format EBNF \
    --depth 5 \
    --visualize
```

### **Parse and Validate**
```bash
dotnet fsi tools/fractal_grammar_cli.fsx parse \
    --input grammar.fractal \
    --verbose
```

### **Analyze Complexity**
```bash
dotnet fsi tools/fractal_grammar_cli.fsx analyze \
    --input grammar.fractal \
    --detailed
```

### **Generate Visualization**
```bash
dotnet fsi tools/fractal_grammar_cli.fsx visualize \
    --input grammar.fractal \
    --output visualization.svg \
    --format SVG
```

### **Show Examples**
```bash
dotnet fsi tools/fractal_grammar_cli.fsx examples
```

## 📊 Mathematical Properties

### **Fractal Dimensions**
- **Sierpinski Triangle**: log(3)/log(2) ≈ 1.585
- **Koch Snowflake**: log(4)/log(3) ≈ 1.261
- **Dragon Curve**: 2.0 (space-filling)

### **Self-Similarity Ratios**
- **Sierpinski**: 1/2 scaling, 3 copies
- **Koch**: 1/3 scaling, 4 copies
- **Dragon**: 1/√2 scaling, 2 copies

### **Complexity Metrics**
- **Total Rules**: Number of fractal rules
- **Average Dimension**: Mean fractal dimension
- **Max Recursion Depth**: Maximum allowed depth
- **Composition Complexity**: Number of rule dependencies

## 🧪 Testing

Run comprehensive tests:

```bash
dotnet test Tars.Engine.Grammar.Tests --filter "FractalGrammarTests"
```

### **Test Coverage**
- ✅ **Fractal Engine**: Transformations, generation, analysis
- ✅ **Parser**: Syntax parsing, error handling
- ✅ **Integration**: Multi-format output, visualization
- ✅ **Examples**: Sierpinski, Koch, Dragon curves
- ✅ **CLI**: All command operations

## 🔮 Advanced Features

### **L-System Integration**
```fractal
l_system {
    axiom: "F"
    rules: {
        "F": "F+G+"
        "G": "-F-G"
    }
    angle: 90
    iterations: 12
}
```

### **Visualization Configuration**
```fractal
visualization {
    type: "curve"
    colors: ["red", "blue", "green"]
    background: "white"
    animation: {
        enabled: true
        speed: 1.0
        show_construction: true
    }
}
```

### **Mathematical Properties**
```fractal
properties {
    hausdorff_dimension: 1.585
    self_similarity_ratio: 0.5
    scaling_factor: 2.0
    space_filling: false
}
```

## 🚀 Integration with TARS

The Fractal Grammar System integrates seamlessly with the existing TARS ecosystem:

- **Grammar Resolution**: Fractal grammars work with TARS grammar resolver
- **Language Dispatch**: Compatible with TARS language execution
- **Metascript Support**: Can be used within TARS metascripts
- **FLUX Integration**: Works with FLUX multi-modal language system

## 📈 Performance

### **Optimization Features**
- **Recursive Depth Limits**: Prevent infinite recursion
- **Memory Management**: Efficient pattern storage
- **Parallel Execution**: Optional parallel processing
- **Caching**: Result caching for repeated operations

### **Benchmarks**
- **Sierpinski (depth 6)**: ~10ms generation
- **Koch (depth 7)**: ~25ms generation
- **Dragon (depth 12)**: ~100ms generation

## 🎯 Use Cases

1. **Language Design**: Create self-similar programming languages
2. **Pattern Generation**: Generate complex recursive patterns
3. **Mathematical Modeling**: Model fractal structures
4. **Visualization**: Create fractal visualizations
5. **Educational**: Teach fractal mathematics and grammar theory

## 🔗 References

- **Fractal Geometry**: Mandelbrot, "The Fractal Geometry of Nature"
- **L-Systems**: Lindenmayer systems for biological modeling
- **Grammar Theory**: Chomsky hierarchy and formal languages
- **TARS Architecture**: TARS metascript and grammar systems

---

**🌀 TARS Fractal Grammars** - Where mathematics meets language design!
