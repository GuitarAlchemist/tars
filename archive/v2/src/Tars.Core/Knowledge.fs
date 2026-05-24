namespace Tars.Core

open System
open System.IO
open System.Text.RegularExpressions

/// <summary>
/// TARS Knowledge Base - A simple markdown-based wiki for storing beliefs,
/// learned procedures, and facts. Human-readable and version-controlled.
/// </summary>
module Knowledge =

    /// Categories of knowledge entries
    type Category =
        | Beliefs       // Things TARS believes to be true
        | Learned       // Procedures/patterns learned through experience
        | Facts         // Verified factual information
        | Meta          // Knowledge about TARS itself
        | Custom of string

    /// Confidence level for beliefs
    type Confidence =
        | High          // Verified or strongly supported
        | Medium        // Reasonable evidence
        | Low           // Tentative or uncertain
        | Unknown

    /// Source of knowledge
    type Source =
        | Observed      // Learned from direct observation
        | Told          // User or external source provided it
        | Inferred      // Derived from other knowledge
        | Evolved       // Learned during evolution/self-improvement

    /// Kind of pattern stored in grammar field
    type PatternKind =
        | PromptPattern     // Successful prompt structures
        | OutputPattern     // Expected output formats
        | AntiPattern       // Known failure modes
        | ReasoningPattern  // Chain-of-thought templates

    /// A knowledge entry with metadata
    type Entry = {
        Id: string
        Title: string
        Category: Category
        Content: string
        Grammar: string option      // Distilled grammar pattern (e.g., "<answer>{text}</answer>")
        PatternKind: PatternKind option  // What kind of pattern the grammar represents
        Confidence: Confidence
        Source: Source
        Tags: string list
        CreatedAt: DateTime
        UpdatedAt: DateTime
    }

    /// Parse front matter from markdown content
    let private parseFrontMatter (content: string) : Map<string, string> * string =
        let pattern = @"^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$"
        let m = Regex.Match(content, pattern)
        if m.Success then
            let frontMatter = m.Groups.[1].Value
            let body = m.Groups.[2].Value
            let pairs =
                frontMatter.Split('\n')
                |> Array.choose (fun line ->
                    let parts = line.Split(':', 2)
                    if parts.Length = 2 then
                        Some (parts.[0].Trim(), parts.[1].Trim())
                    else None)
                |> Map.ofArray
            (pairs, body)
        else
            (Map.empty, content)

    /// Generate front matter for an entry
    let private toFrontMatter (entry: Entry) : string =
        let categoryStr =
            match entry.Category with
            | Beliefs -> "beliefs"
            | Learned -> "learned"
            | Facts -> "facts"
            | Meta -> "meta"
            | Custom s -> s
        let confidenceStr =
            match entry.Confidence with
            | High -> "high" | Medium -> "medium" | Low -> "low" | Unknown -> "unknown"
        let sourceStr =
            match entry.Source with
            | Observed -> "observed" | Told -> "told" | Inferred -> "inferred" | Evolved -> "evolved"
        let patternKindStr =
            match entry.PatternKind with
            | Some PromptPattern -> "prompt"
            | Some OutputPattern -> "output"
            | Some AntiPattern -> "anti"
            | Some ReasoningPattern -> "reasoning"
            | None -> ""
        let grammarLine =
            match entry.Grammar with
            | Some g -> $"grammar: {g}\n"
            | None -> ""
        let patternLine =
            match entry.PatternKind with
            | Some _ -> $"pattern_kind: {patternKindStr}\n"
            | None -> ""

        $"""---
id: {entry.Id}
title: {entry.Title}
category: {categoryStr}
confidence: {confidenceStr}
source: {sourceStr}
tags: {String.Join(", ", entry.Tags)}
{grammarLine}{patternLine}created: {entry.CreatedAt:O}
updated: {entry.UpdatedAt:O}
---

{entry.Content}"""

    /// Parse category from string
    let private parseCategory (s: string) : Category =
        match s.ToLowerInvariant() with
        | "beliefs" -> Beliefs | "learned" -> Learned
        | "facts" -> Facts | "meta" -> Meta
        | other -> Custom other

    /// Parse confidence from string
    let private parseConfidence (s: string) : Confidence =
        match s.ToLowerInvariant() with
        | "high" -> High | "medium" -> Medium | "low" -> Low | _ -> Unknown

    /// Parse source from string
    let private parseSource (s: string) : Source =
        match s.ToLowerInvariant() with
        | "observed" -> Observed | "told" -> Told | "inferred" -> Inferred | "evolved" -> Evolved | _ -> Observed

    /// Parse pattern kind from string
    let private parsePatternKind (s: string) : PatternKind option =
        match s.ToLowerInvariant() with
        | "prompt" -> Some PromptPattern
        | "output" -> Some OutputPattern
        | "anti" -> Some AntiPattern
        | "reasoning" -> Some ReasoningPattern
        | _ -> None

    /// Parse an entry from file content
    let parseEntry (filePath: string) (content: string) : Entry option =
        let (meta, body) = parseFrontMatter content
        let get key def = meta |> Map.tryFind key |> Option.defaultValue def
        let getOpt key = meta |> Map.tryFind key |> Option.filter (fun s -> s.Trim() <> "")
        try
            Some {
                Id = get "id" (Path.GetFileNameWithoutExtension(filePath))
                Title = get "title" (Path.GetFileNameWithoutExtension(filePath))
                Category = parseCategory (get "category" "learned")
                Content = body.Trim()
                Grammar = getOpt "grammar"
                PatternKind = getOpt "pattern_kind" |> Option.bind parsePatternKind
                Confidence = parseConfidence (get "confidence" "unknown")
                Source = parseSource (get "source" "observed")
                Tags = (get "tags" "").Split(',') |> Array.map (fun s -> s.Trim()) |> Array.filter ((<>) "") |> Array.toList
                CreatedAt = get "created" "" |> fun s -> match DateTime.TryParse(s) with | true, d -> d | _ -> DateTime.UtcNow
                UpdatedAt = get "updated" "" |> fun s -> match DateTime.TryParse(s) with | true, d -> d | _ -> DateTime.UtcNow
            }
        with _ -> None

    /// Knowledge base backed by a directory of markdown files
    type KnowledgeBase(basePath: string) =
        let ensureDir () =
            if not (Directory.Exists basePath) then
                Directory.CreateDirectory basePath |> ignore
            ["beliefs"; "learned"; "facts"; "meta"]
            |> List.iter (fun sub ->
                let p = Path.Combine(basePath, sub)
                if not (Directory.Exists p) then Directory.CreateDirectory p |> ignore)

        let categoryToFolder = function
            | Beliefs -> "beliefs" | Learned -> "learned"
            | Facts -> "facts" | Meta -> "meta" | Custom s -> s

        let getFilePath (entry: Entry) =
            let folder = categoryToFolder entry.Category
            let sanitized = Regex.Replace(entry.Id, @"[^\w\-]", "_")
            Path.Combine(basePath, folder, $"{sanitized}.md")

        do ensureDir()

        /// List all entries
        member _.List(?category: Category) : Entry list =
            let folders =
                match category with
                | Some c -> [Path.Combine(basePath, categoryToFolder c)]
                | None -> Directory.GetDirectories(basePath) |> Array.toList
            folders
            |> List.collect (fun folder ->
                if Directory.Exists folder then
                    Directory.GetFiles(folder, "*.md")
                    |> Array.choose (fun f ->
                        try File.ReadAllText(f) |> parseEntry f
                        with _ -> None)
                    |> Array.toList
                else [])

        /// Get a specific entry by ID
        member this.Get(id: string) : Entry option =
            this.List() |> List.tryFind (fun e -> e.Id = id)

        /// Add or update an entry
        member _.Save(entry: Entry) : unit =
            ensureDir()
            let path = getFilePath entry
            let dir = Path.GetDirectoryName(path)
            if not (Directory.Exists dir) then Directory.CreateDirectory dir |> ignore
            File.WriteAllText(path, toFrontMatter entry)

        /// Create a new entry
        member this.Add(title: string, content: string, category: Category, ?confidence: Confidence, ?source: Source, ?tags: string list, ?grammar: string, ?patternKind: PatternKind) : Entry =
            let entry = {
                Id = Guid.NewGuid().ToString("N")[..7]
                Title = title
                Category = category
                Content = content
                Grammar = grammar
                PatternKind = patternKind
                Confidence = defaultArg confidence Unknown
                Source = defaultArg source Observed
                Tags = defaultArg tags []
                CreatedAt = DateTime.UtcNow
                UpdatedAt = DateTime.UtcNow
            }
            this.Save(entry)
            entry

        /// Delete an entry
        member _.Delete(id: string) : bool =
            let files =
                Directory.GetDirectories(basePath)
                |> Array.collect (fun d -> Directory.GetFiles(d, "*.md"))
            files
            |> Array.tryFind (fun f ->
                match File.ReadAllText(f) |> parseEntry f with
                | Some e -> e.Id = id
                | None -> false)
            |> Option.map (fun f -> File.Delete(f); true)
            |> Option.defaultValue false

        /// Search entries by keyword
        member this.Search(query: string) : Entry list =
            let q = query.ToLowerInvariant()
            this.List()
            |> List.filter (fun e ->
                e.Title.ToLowerInvariant().Contains(q) ||
                e.Content.ToLowerInvariant().Contains(q) ||
                e.Tags |> List.exists (fun t -> t.ToLowerInvariant().Contains(q)))

        /// Get the base path
        member _.BasePath = basePath

