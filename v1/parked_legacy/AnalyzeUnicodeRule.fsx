#r "./Tars.Engine.Grammar/bin/Release/net9.0/Tars.Engine.Grammar.dll"
open Tars.Engine.Grammar
open Tars.Engine.Grammar.GrammarResolver

let unicode =
    generateGrammarFromExamples "JsonUnicode" [
        "{\"emoji\":\"😀\",\"math\":\"∑\"}"
        "{\"emoji\":\"雪\",\"math\":\"λ\"}"
    ]

let lines = unicode.Content.Split('\n')
let ruleLine = lines |> Array.find (fun l -> l.Contains "jsonunicode_example_01 =")
printfn "%s" ruleLine
printfn "%A" (ruleLine |> Seq.map (fun ch -> int ch) |> Seq.toArray)
