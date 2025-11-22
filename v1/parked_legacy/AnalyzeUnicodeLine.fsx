#r "./Tars.Engine.Grammar/bin/Release/net9.0/Tars.Engine.Grammar.dll"
open Tars.Engine.Grammar
open Tars.Engine.Grammar.GrammarResolver

let unicode =
    generateGrammarFromExamples "JsonUnicode" [
        "{\"emoji\":\"😀\",\"math\":\"∑\"}"
        "{\"emoji\":\"雪\",\"math\":\"λ\"}"
    ]

let line =
    unicode.Content.Split('\n')
    |> Array.find (fun l -> l.Contains "jsonunicode_example_01")

printfn "%s" line
printfn "%A" (line |> Seq.map int |> Seq.toArray)
