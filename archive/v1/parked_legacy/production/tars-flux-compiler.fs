// TARS FLUX Pattern Compiler
// Compiles roadmap FLUX patterns to F# implementations

module TarsFluxCompiler =
    let compileResultPattern() = "type TarsError = ValidationError of string"
    let compileDocumentationPattern functionName = sprintf "/// <summary>\n/// %s\n/// </summary>" functionName