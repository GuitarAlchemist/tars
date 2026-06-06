namespace Tars.Metascript

open System
open System.IO
open System.Text.RegularExpressions
open Tars.Core
open Tars.Core.WorkflowOfThought

/// <summary>
/// Parser for the .trsx (Workflow of Thought) DSL.
/// Format:
/// GOAL "Description"
/// NODE name [REASON|WORK] operation [params]
/// EDGE from -> to
/// </summary>
module TrsxParser =

    let parse (content: string) : WorkflowGraph =
        let lines = content.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
        let mutable goal = "Unnamed Workflow"
        let mutable nodes = Map.empty<NodeId, WotNode>
        let mutable nodeIds = Map.empty<string, NodeId>
        let mutable edges = []

        for line in
            lines
            |> Array.map (fun l -> l.Trim())
            |> Array.filter (fun l -> not (l.StartsWith("#") || String.IsNullOrWhiteSpace(l))) do
            let goalMatch = Regex.Match(line, "GOAL\s+\"?([^\"]+)\"?", RegexOptions.IgnoreCase)

            let nodeMatch =
                Regex.Match(line, "NODE\s+\"?([^\"]+)\"?\s+(REASON|WORK)\s+(\w+)(.*)", RegexOptions.IgnoreCase)

            let edgeMatch =
                Regex.Match(line, "EDGE\s+\"?([^\"]+)\"?\s*->\s*\"?([^\"]+)\"?", RegexOptions.IgnoreCase)

            if goalMatch.Success then
                goal <- goalMatch.Groups.[1].Value.Trim().Trim('"')

            elif nodeMatch.Success then
                let name = nodeMatch.Groups.[1].Value.Trim().Trim('"')
                let type' = nodeMatch.Groups.[2].Value.ToUpper()
                let opName = nodeMatch.Groups.[3].Value
                let opParam = nodeMatch.Groups.[4].Value.Trim()

                // Strip trailing tags like /> or </NODE> if they exist
                let cleanParam =
                    if opParam.EndsWith("/>") then
                        opParam.Substring(0, opParam.Length - 2).Trim()
                    else
                        Regex.Replace(opParam, "</NODE>$", "", RegexOptions.IgnoreCase).Trim()

                let nid = NodeId.create ()
                nodeIds <- Map.add name nid nodeIds

                let node =
                    if type' = "REASON" then
                        let op =
                            match opName.ToLower() with
                            | "plan" -> ReasonOperation.Plan cleanParam
                            | "explain" -> ReasonOperation.Explain cleanParam
                            | _ -> ReasonOperation.Plan cleanParam

                        Reason
                            { Id = nid
                              Name = name
                              Operation = op
                              Input = NodeContent.empty
                              Output = None
                              Model = None
                              ModelHint = None
                              Budget = NodeBudget.default'
                              Policy = PolicyGate.permissive
                              Evidence = None }
                    else
                        let op =
                            match opName.ToLower() with
                            | "toolcall" ->
                                let toolMatch = Regex.Match(cleanParam, "\"?([^\"]+)\"?\s*(.*)")

                                if toolMatch.Success then
                                    let toolName = toolMatch.Groups.[1].Value
                                    let jsonStr = toolMatch.Groups.[2].Value.Trim()

                                    let args =
                                        if String.IsNullOrWhiteSpace(jsonStr) then
                                            Map.empty
                                        else
                                            try
                                                let doc = System.Text.Json.JsonDocument.Parse(jsonStr)

                                                doc.RootElement.EnumerateObject()
                                                |> Seq.map (fun prop ->
                                                    (prop.Name, box (prop.Value.GetRawText().Trim('"'))))
                                                |> Map.ofSeq
                                            with _ ->
                                                Map.empty

                                    WorkOperation.ToolCall(toolName, args)
                                else
                                    WorkOperation.ToolCall(cleanParam.Trim('"'), Map.empty)
                            | "verify" -> WorkOperation.Verify(cleanParam, "success")
                            | _ -> WorkOperation.Fetch cleanParam

                        Work
                            { Id = nid
                              Name = name
                              Operation = op
                              Input = NodeContent.empty
                              Output = None
                              Budget = NodeBudget.minimal
                              Policy = PolicyGate.permissive
                              Evidence = None
                              SideEffects = [] }

                nodes <- Map.add nid node nodes

            elif edgeMatch.Success then
                let fromName = edgeMatch.Groups.[1].Value.Trim().Trim('"')
                let toName = edgeMatch.Groups.[2].Value.Trim().Trim('"')

                match Map.tryFind fromName nodeIds, Map.tryFind toName nodeIds with
                | Some fid, Some tid -> edges <- edges @ [ (tid, DependsOn, fid) ]
                | _ -> ()

        // Identify entry point: node(s) with no incoming DependsOn edges
        let allNodes = nodes.Keys |> Set.ofSeq
        let targets = edges |> List.map (fun (to', rel, from') -> to') |> Set.ofList
        let sources = Set.difference allNodes targets

        let entryPoint =
            if sources.IsEmpty then
                if nodes.IsEmpty then
                    NodeId.create ()
                else
                    nodes.Keys |> Seq.head
            else
                sources |> Seq.head

        { Id = Guid.NewGuid()
          Name = goal
          Description = goal
          Nodes = nodes
          Edges = edges
          EntryPoint = entryPoint
          ExitPoints = []
          GlobalBudget = Budget.default'
          GlobalPolicy = PolicyGate.permissive
          AuditLog = [] }

    let loadFile (path: string) : WorkflowGraph =
        let content = File.ReadAllText(path)
        parse content
