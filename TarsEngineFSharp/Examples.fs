namespace Tars.DSL

open System
open System.IO
open Tars.DSL
open Tars.DSL.DataSources
open Tars.DSL.AsyncExecution
open Tars.DSL.PromptEngine
open Tars.DSL.TarsDsl

/// Module containing example usages of the TARS DSL
module Examples =
    /// Type for CSV row parsing
    type Person = {
        Name: string
        City: string
        Age: int
    }

    /// Simple example demonstrating basic data loading and prompt operations
    let basicExample() = async {
        // Create a sample data file
        let sampleDataPath = Path.Combine(Environment.CurrentDirectory, "sample_data.txt")
        File.WriteAllText(sampleDataPath, "This is sample data for TARS to process.")
        
        // Define the workflow using the TARS DSL
        let! result = tars {
            // Load data from file
            let! fileContent = tars.FileData(sampleDataPath, id)
            printfn $"Loaded file content: %s{fileContent}"
            
            // Generate a summary using AI
            let! summary = tars.Summarize fileContent
            printfn $"Summary: %s{summary.Content} (Confidence: %.2f{summary.Confidence})"
            
            // Generate additional content based on the summary
            let! additionalContent = tars.Generate $"Expand on this summary: {summary.Content}"
            printfn $"Additional content: %s{additionalContent.Content}"
            
            return "Workflow completed successfully"
        }
        
        printfn $"Final result: %s{result}"
        
        // Clean up
        if File.Exists(sampleDataPath) then
            File.Delete(sampleDataPath)
    }
    
    /// Example demonstrating alternative syntax
    let alternativeSyntaxExample() = async {
        // Create a sample CSV file
        let sampleDataPath = Path.Combine(Environment.CurrentDirectory, "sample_data.csv")
        File.WriteAllText(sampleDataPath, "Name,City,Age\nJohn,New York,30\nJane,San Francisco,25\nBob,Chicago,40")
        
        // Row parser function
        let rowParser (line: string) =
            let parts = line.Split(',')
            if parts.Length = 3 then
                { Name = parts.[0]; City = parts.[1]; Age = Int32.Parse(parts.[2]) }
            else
                { Name = "Unknown"; City = "Unknown"; Age = 0 }
        
        // Define the workflow using the alternative TARS DSL syntax
        let! result = async {
            // Load data from CSV file
            let! csvData = DATA.CSV sampleDataPath rowParser ','
            printfn $"Loaded CSV data with %d{csvData.Length} rows"
            
            // Perform a web search
            let! searchResults = DATA.WEB_SEARCH "latest AI research"
            printfn $"Search results: %A{searchResults}"
            
            // Analyze the combined data
            let combinedData = 
                csvData 
                |> Array.map (fun row -> $"{row.Name} from {row.City}")
                |> String.concat ", "
                |> fun s -> s + "\n\nSearch results: " + (String.concat ", " searchResults)
            
            let! analysis = AI.ANALYZE combinedData
            printfn $"Analysis: %s{analysis.Content}"
            
            return "Alternative syntax workflow completed successfully"
        }
        
        printfn $"Final result: %s{result}"
        
        // Clean up
        if File.Exists(sampleDataPath) then
            File.Delete(sampleDataPath)
    }
    
    /// Example demonstrating async task execution
    let asyncTaskExample() = async {
        // Define the TARS workflow
        let! result = tars {
            // Define multiple async tasks
            let task1 = async {
                do! Async.Sleep 1000
                return "Task 1 result"
            }
            
            let task2 = async {
                do! Async.Sleep 2000
                return "Task 2 result"
            }
            
            // Execute tasks in parallel
            let! (taskInfo1, result1) = tars.ExecuteTask("Task 1", task1)
            let! (taskInfo2, result2) = tars.ExecuteTask("Task 2", task2)
            
            printfn $"Started tasks: %A{taskInfo1.Id} and %A{taskInfo2.Id}"
            
            // Wait for both tasks to complete
            let! completedTask1 = tars.WaitForTask(taskInfo1.Id, TimeSpan.FromSeconds(5.0))
            let! completedTask2 = tars.WaitForTask(taskInfo2.Id, TimeSpan.FromSeconds(5.0))
            
            printfn $"Task 1 completed: %s{result1}"
            printfn $"Task 2 completed: %s{result2}"
            
            return "Async task workflow completed successfully"
        }
        
        printfn $"Final result: %s{result}"
    }