
// TARS-improved code with functional patterns
module DataProcessor =
    
    type ProcessingResult = 
        | Success of string list
        | Error of string
    
    type LogEntry = 
        | ErrorFound of string
        | LongString of string  
        | ShortString of string
    
    let private logEntry (entry: LogEntry) : unit =
        match entry with
        | ErrorFound msg -> printfn "Error found: %s" msg
        | LongString str -> printfn "Long string: %s" str
        | ShortString str -> printfn "Short string: %s" str
    
    let private processItem (item: string) : LogEntry option =
        if String.IsNullOrEmpty(item) then None
        elif item.Contains("error") then Some (ErrorFound item)
        elif item.Length > 10 then Some (LongString item)
        else Some (ShortString item)
    
    let processData (input: string array) : ProcessingResult =
        try
            match input with
            | null -> Error "Input cannot be null"
            | [||] -> Success []
            | items ->
                items
                |> Array.choose processItem
                |> Array.iter logEntry
                
                let processedItems = 
                    items 
                    |> Array.filter (not << String.IsNullOrEmpty)
                    |> Array.toList
                
                Success processedItems
        with
        | ex -> Error $"Processing failed: {ex.Message}"
