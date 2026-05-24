module Tars.Interface.Cli.Commands.TestGrammar

open System
open Tars.Cortex.Grammar

let run (file: string) =
    try
        let content = System.IO.File.ReadAllText(file)
        let goals = Parser.parse content
        printfn "Parsed %d goals" goals.Length

        for g in goals do
            printfn "Goal: %s" g.Name

            for t in g.Tasks do
                printfn "  - Task: %s" t.Name

        0
    with ex ->
        printfn "Error: %s" ex.Message
        1
