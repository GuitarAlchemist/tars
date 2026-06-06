namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.VectorStore

type DemoCommand(logger: ILogger<DemoCommand>, vectorStore: IVectorStore) = 
    interface ICommand with
        member self.Name = "demo-semantic-search"
        member self.Description = "Demonstrates a real-world semantic search using the TARS vector store."
        member self.Usage = "tars demo-semantic-search \"<query>\""
        member self.Examples = ["tars demo-semantic-search \"What is functional programming?\""]

        member self.ValidateOptions(options: CommandOptions) =
            not options.Arguments.IsEmpty

        member self.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    let query = options.Arguments.[0]

                    // Add documents to the vector store
                    let documents = 
                        [
                            { Id = "1"; Content = "Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing-state and mutable data."; Embedding = null; Tags = ["programming"]; Timestamp = DateTime.UtcNow; Source = None }
                            { Id = "2"; Content = "Object-oriented programming is a programming paradigm based on the concept of 'objects', which can contain data and code: data in the form of fields, and code, in the form of procedures."; Embedding = null; Tags = ["programming"]; Timestamp = DateTime.UtcNow; Source = None }
                            { Id = "3"; Content = "The TARS project is a powerful AI-driven development and automation system."; Embedding = null; Tags = ["ai"]; Timestamp = DateTime.UtcNow; Source = None }
                        ]
                    do! vectorStore.AddDocuments(documents)

                    // Search for the most similar document
                    let! searchResults = vectorStore.Search({ Text = query; Embedding = null; Filters = Map.empty; MaxResults = 1; MinScore = 0.5 })

                    // Print the result
                    match searchResults with
                    | [] -> printfn "No results found."
                    | result :: _ ->
                        printfn "Best match: %s" result.Document.Content
                        printfn "Similarity score: %f" result.FinalScore

                    return CommandResult.success("Semantic search demo completed successfully.")
                with
                | ex -> 
                    logger.LogError(ex, "Error executing semantic search demo")
                    return CommandResult.failure(ex.Message)
            }