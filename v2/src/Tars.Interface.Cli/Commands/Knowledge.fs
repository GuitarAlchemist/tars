module Tars.Interface.Cli.Commands.Knowledge

open System
open System.IO
open Tars.Core.Knowledge

type KnowledgeOptions =
    { Command: string // list, add, search, show, delete
      Query: string option
      Title: string option
      Content: string option
      Category: string option
      Tags: string option }

let private getKnowledgePath () =
    // Look for knowledge folder in .tars directory
    let candidates =
        [ Path.Combine(Environment.CurrentDirectory, ".tars", "knowledge")
          Path.Combine(Environment.CurrentDirectory, "v2", ".tars", "knowledge")
          Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", ".tars", "knowledge") ]

    candidates
    |> List.tryFind Directory.Exists
    |> Option.defaultValue candidates.[0]

let private printEntry (entry: Entry) =
    let catStr =
        match entry.Category with
        | Beliefs -> "beliefs"
        | Learned -> "learned"
        | Facts -> "facts"
        | Meta -> "meta"
        | Custom s -> s

    let confStr =
        match entry.Confidence with
        | High -> "high"
        | Medium -> "medium"
        | Low -> "low"
        | Unknown -> "?"

    Console.ForegroundColor <- ConsoleColor.Cyan
    printf "[%s] " entry.Id
    Console.ForegroundColor <- ConsoleColor.White
    printf "%s " entry.Title
    Console.ForegroundColor <- ConsoleColor.DarkGray
    printfn "(%s, %s)" catStr confStr
    Console.ResetColor()

let private printEntryFull (entry: Entry) =
    printEntry entry
    Console.ForegroundColor <- ConsoleColor.DarkGray
    printfn "Tags: %s" (String.Join(", ", entry.Tags))

    printfn
        "Created: %s | Updated: %s"
        (entry.CreatedAt.ToString("yyyy-MM-dd"))
        (entry.UpdatedAt.ToString("yyyy-MM-dd"))

    Console.ResetColor()
    printfn ""
    printfn "%s" entry.Content
    printfn ""

let private parseCategory (s: string) : Category =
    match s.ToLowerInvariant() with
    | "beliefs"
    | "belief" -> Beliefs
    | "learned"
    | "learn" -> Learned
    | "facts"
    | "fact" -> Facts
    | "meta" -> Meta
    | other -> Custom other

let run (options: KnowledgeOptions) =
    let kb = KnowledgeBase(getKnowledgePath ())

    match options.Command.ToLowerInvariant() with
    | "list"
    | "ls" ->
        let category = options.Category |> Option.map parseCategory
        let entries = kb.List(?category = category)

        if entries.IsEmpty then
            printfn "No knowledge entries found."
        else
            printfn "📚 Knowledge Base (%d entries)\n" entries.Length
            entries |> List.iter printEntry

    | "search"
    | "find" ->
        match options.Query with
        | Some q ->
            let results = kb.Search(q)

            if results.IsEmpty then
                printfn "No entries matching '%s'" q
            else
                printfn "🔍 Search results for '%s' (%d found)\n" q results.Length
                results |> List.iter printEntry
        | None -> printfn "Usage: tars knowledge search <query>"

    | "show"
    | "get" ->
        match options.Query with
        | Some id ->
            match kb.Get(id) with
            | Some entry -> printEntryFull entry
            | None -> printfn "Entry '%s' not found" id
        | None -> printfn "Usage: tars knowledge show <id>"

    | "add"
    | "new" ->
        match options.Title, options.Content with
        | Some title, Some content ->
            let category =
                options.Category |> Option.map parseCategory |> Option.defaultValue Learned

            let tags =
                options.Tags
                |> Option.map (fun t -> t.Split(',') |> Array.map (_.Trim()) |> Array.toList)
                |> Option.defaultValue []

            let entry = kb.Add(title, content, category, tags = tags, source = Told)
            Console.ForegroundColor <- ConsoleColor.Green
            printfn "✓ Created entry: %s" entry.Id
            Console.ResetColor()
        | _ ->
            printfn
                "Usage: tars knowledge add --title \"Title\" --content \"Content\" [--category beliefs|learned|facts|meta] [--tags \"tag1,tag2\"]"

    | "delete"
    | "rm" ->
        match options.Query with
        | Some id ->
            if kb.Delete(id) then
                Console.ForegroundColor <- ConsoleColor.Yellow
                printfn "✓ Deleted entry: %s" id
                Console.ResetColor()
            else
                printfn "Entry '%s' not found" id
        | None -> printfn "Usage: tars knowledge delete <id>"

    | "path" -> printfn "Knowledge base path: %s" kb.BasePath

    | _ ->
        printfn "TARS Knowledge Base\n"
        printfn "Commands:"
        printfn "  list [--category <cat>]     List all entries"
        printfn "  search <query>              Search entries"
        printfn "  show <id>                   Show entry details"
        printfn "  add --title --content       Add new entry"
        printfn "  delete <id>                 Delete entry"
        printfn "  path                        Show knowledge base path"
