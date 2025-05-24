namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Module for processing exploration files
/// </summary>
module ExplorationFileProcessor =
    /// <summary>
    /// Gets all exploration files in the specified directory
    /// </summary>
    let getExplorationFiles (directoryPath: string) (maxFiles: int) =
        if Directory.Exists(directoryPath) then
            Directory.GetFiles(directoryPath, "*.md", SearchOption.AllDirectories)
            |> Array.filter (fun file -> not (file.Contains("README.md")))
            |> Array.take (min maxFiles (Array.length (Directory.GetFiles(directoryPath, "*.md", SearchOption.AllDirectories))))
            |> Array.toList
        else
            []
            
    /// <summary>
    /// Determines the source type based on the file path
    /// </summary>
    let determineSourceType (filePath: string) =
        if filePath.Contains("/Chats/") || filePath.Contains("\\Chats\\") then
            KnowledgeSourceType.Chat
        elif filePath.Contains("/Reflections/") || filePath.Contains("\\Reflections\\") then
            KnowledgeSourceType.Reflection
        elif filePath.Contains("/features/") || filePath.Contains("\\features\\") then
            KnowledgeSourceType.Feature
        elif filePath.Contains("/architecture/") || filePath.Contains("\\architecture\\") then
            KnowledgeSourceType.Architecture
        elif filePath.Contains("/tutorials/") || filePath.Contains("\\tutorials\\") then
            KnowledgeSourceType.Tutorial
        else
            KnowledgeSourceType.Documentation
            
    /// <summary>
    /// Reads the content of a file
    /// </summary>
    let readFileContent (filePath: string) =
        task {
            try
                return! File.ReadAllTextAsync(filePath)
            with
            | ex -> 
                return sprintf "Error reading file: %s" ex.Message
        }
