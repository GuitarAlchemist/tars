#r "./Tars.Engine.Grammar/bin/Release/net9.0/Tars.Engine.Grammar.dll"
open Tars.Engine.Grammar
open Tars.Engine.Grammar.GrammarResolver

let unicode =
    generateGrammarFromExamples "JsonUnicode" [
        "{\"emoji\":\"😀\",\"math\":\"∑\"}"
        "{\"emoji\":\"雪\",\"math\":\"λ\"}"
    ]

printfn "Contains 😀: %b" (unicode.Content.Contains "😀")
printfn "Content chars: %A" (unicode.Content |> Seq.takeWhile (fun c -> c <> '\n') |> Seq.toArray)
