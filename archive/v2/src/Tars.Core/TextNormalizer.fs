namespace Tars.Core

open System
open System.Text.RegularExpressions

/// Utilities for normalizing and preprocessing text for NLP tasks
module TextNormalizer =

    /// Common English stop words
    let private stopWords =
        Set.ofList
            [ "a"
              "an"
              "and"
              "are"
              "as"
              "at"
              "be"
              "by"
              "for"
              "from"
              "has"
              "he"
              "in"
              "is"
              "it"
              "its"
              "of"
              "on"
              "that"
              "the"
              "to"
              "was"
              "were"
              "will"
              "with"
              "this"
              "but"
              "they"
              "have"
              "had"
              "what"
              "when"
              "where"
              "who"
              "which"
              "why"
              "how" ]

    /// Normalize text: lowercase, remove special characters, collapse whitespace
    let normalize (text: string) =
        if String.IsNullOrWhiteSpace(text) then
            ""
        else
            text.ToLowerInvariant()
            |> fun s -> Regex.Replace(s, @"[^a-z0-9\s]", "") // Keep only alphanumeric and space
            |> fun s -> Regex.Replace(s, @"\s+", " ") // Collapse multiple spaces
            |> fun s -> s.Trim()

    /// Split text into tokens (words)
    let tokenize (text: string) =
        if String.IsNullOrWhiteSpace(text) then
            []
        else
            text.Split(' ', StringSplitOptions.RemoveEmptyEntries) |> Array.toList

    /// Remove stop words from a list of tokens
    let removeStopWords (tokens: string list) =
        tokens |> List.filter (fun t -> not (stopWords.Contains(t)))

    /// Extract keywords from text (normalize -> tokenize -> remove stop words -> dedup)
    let extractKeywords (text: string) =
        text |> normalize |> tokenize |> removeStopWords |> List.distinct
