#r "./Tars.Engine.Grammar/bin/Release/net9.0/Tars.Engine.Grammar.dll"

open System
open Tars.Engine.Grammar

let rules = [
    "RFC3986", "URI-reference";
    "RFC3986", "hier-part";
    "RFC3986", "authority";
    "RFC5234", "HEXDIG";
    "RFC9110", "field-name"
]

for (rfc, rule) in rules do
    try
        let source = GrammarSource.EmbeddedRFC(rfc, rule)
        let exists = GrammarSource.exists source
        printfn "%s:%s exists=%b" rfc rule exists
        if exists then
            let content = GrammarSource.getContent source
            content.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            |> Array.truncate 3
            |> String.concat "\n"
            |> printfn "%s\n"
    with ex ->
        printfn "%s:%s error %s" rfc rule ex.Message
