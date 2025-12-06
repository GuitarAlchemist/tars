namespace Tars.Tools.Standard

open System
open System.IO
open Tars.Tools

module MetascriptTools =

    /// Simple metascript parser for embedded scripts
    let private parseMetascript (script: string) =
        let lines =
            script.Split('\n')
            |> Array.map (fun s -> s.Trim())
            |> Array.filter (fun s -> s.Length > 0 && not (s.StartsWith("//")))

        let steps = ResizeArray<string * string * string>()

        for line in lines do
            if line.StartsWith("DEFINE ", StringComparison.OrdinalIgnoreCase) then
                let rest = line.Substring(7).Trim()
                let colonIdx = rest.IndexOf(':')

                if colonIdx > 0 then
                    let name = rest.Substring(0, colonIdx).Trim()
                    let value = rest.Substring(colonIdx + 1).Trim().Trim('"')
                    steps.Add(("DEFINE", name, value))
            elif line.StartsWith("EXECUTE ", StringComparison.OrdinalIgnoreCase) then
                let rest = line.Substring(8).Trim()
                let colonIdx = rest.IndexOf(':')

                if colonIdx > 0 then
                    let tool = rest.Substring(0, colonIdx).Trim()
                    let args = rest.Substring(colonIdx + 1).Trim()
                    steps.Add(("EXECUTE", tool, args))
            elif line.StartsWith("SET ", StringComparison.OrdinalIgnoreCase) then
                let rest = line.Substring(4).Trim()
                let eqIdx = rest.IndexOf('=')

                if eqIdx > 0 then
                    let varName = rest.Substring(0, eqIdx).Trim()
                    let varValue = rest.Substring(eqIdx + 1).Trim()
                    steps.Add(("SET", varName, varValue))
            elif line.StartsWith("PRINT ", StringComparison.OrdinalIgnoreCase) then
                let message = line.Substring(6).Trim().Trim('"')
                steps.Add(("PRINT", "", message))

        steps |> Seq.toList

    [<TarsToolAttribute("parse_metascript",
                        "Parses a TARS metascript and returns the steps. Input: metascript code as string.")>]
    let parseMetascriptTool (script: string) =
        task {
            try
                printfn "PARSING METASCRIPT..."
                let steps = parseMetascript script

                if steps.Length = 0 then
                    return "No valid metascript steps found."
                else
                    let stepList =
                        steps
                        |> List.mapi (fun i (op, name, value) ->
                            let preview =
                                if value.Length > 50 then
                                    value.Substring(0, 50) + "..."
                                else
                                    value

                            sprintf "  %d. %s %s: %s" (i + 1) op name preview)
                        |> String.concat "\n"

                    return sprintf "Parsed %d metascript steps:\n%s" steps.Length stepList
            with ex ->
                return "parse_metascript error: " + ex.Message
        }

    [<TarsToolAttribute("run_metascript", "Runs a simple TARS metascript. Input: metascript code as string.")>]
    let runMetascript (script: string) =
        task {
            try
                printfn "RUNNING METASCRIPT..."
                let steps = parseMetascript script

                if steps.Length = 0 then
                    return "No valid metascript steps to run."
                else
                    let mutable variables = Map.empty<string, string>
                    let output = ResizeArray<string>()

                    for (op, name, value) in steps do
                        match op with
                        | "DEFINE" ->
                            output.Add("DEFINED " + name)
                            variables <- variables.Add(name, value)
                        | "SET" ->
                            output.Add("SET " + name + " = " + value)
                            variables <- variables.Add(name, value)
                        | "PRINT" ->
                            let mutable resolved = value

                            for kvp in variables do
                                let pattern = "{{" + kvp.Key + "}}"
                                resolved <- resolved.Replace(pattern, kvp.Value)

                            output.Add("PRINT: " + resolved)
                            printfn "   %s" resolved
                        | "EXECUTE" -> output.Add("EXECUTE " + name)
                        | _ -> output.Add("Unknown: " + op)

                    let result = String.concat "\n" output
                    printfn "Metascript complete"
                    return sprintf "Metascript execution complete (%d steps):\n%s" steps.Length result
            with ex ->
                return "run_metascript error: " + ex.Message
        }

    [<TarsToolAttribute("create_metascript", "Creates a metascript template for a given task. Input: task description.")>]
    let createMetascript (taskDescription: string) =
        task {
            let preview =
                if taskDescription.Length > 50 then
                    taskDescription.Substring(0, 50)
                else
                    taskDescription

            printfn "CREATING METASCRIPT for: %s..." preview

            let template =
                "// TARS Metascript Template\n"
                + "// Generated for: "
                + taskDescription
                + "\n\n"
                + "DEFINE goal: \""
                + taskDescription
                + "\"\n"
                + "SET status = \"started\"\n"
                + "EXECUTE explore_project: {\"path\": \".\"}\n"
                + "SET status = \"complete\"\n"
                + "PRINT \"Task complete\""

            return
                "Generated Metascript Template:\n\n"
                + template
                + "\n\nUse run_metascript to execute it."
        }
