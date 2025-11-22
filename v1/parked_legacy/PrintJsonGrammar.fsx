#r "./Tars.Engine.Grammar/bin/Release/net9.0/Tars.Engine.Grammar.dll"
open Tars.Engine.Grammar
open Tars.Engine.Grammar.GrammarResolver

let escapes =
    generateGrammarFromExamples "JsonEscapes" [
        "{\"text\":\"Line1\\nLine2\\tTabbed\"}"
        "{\"text\":\"Value with\\r carriage\"}"
    ]
printfn "Escapes:\n%s" escapes.Content

let unicode =
    generateGrammarFromExamples "JsonUnicode" [
        "{\"emoji\":\"😀\",\"math\":\"∑\"}"
        "{\"emoji\":\"雪\",\"math\":\"λ\"}"
    ]
printfn "\nUnicode:\n%s" unicode.Content
