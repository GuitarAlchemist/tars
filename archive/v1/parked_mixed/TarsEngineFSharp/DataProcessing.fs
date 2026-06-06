namespace Tars.DSL

open System

/// Module for data processing operations
module DataProcessing =
    /// Process text data
    let processText (text: string) (processor: string -> string) : string =
        processor text
    
    /// Filter data based on a predicate
    let filterData<'T> (data: 'T[]) (predicate: 'T -> bool) : 'T[] =
        Array.filter predicate data
    
    /// Map data using a transformation function
    let mapData<'T, 'U> (data: 'T[]) (transform: 'T -> 'U) : 'U[] =
        Array.map transform data
    
    /// Aggregate data using a folder function
    let aggregateData<'T, 'State> (data: 'T[]) (initialState: 'State) (folder: 'State -> 'T -> 'State) : 'State =
        Array.fold folder initialState data
    
    /// Join two datasets based on a key selector
    let joinData<'T, 'U, 'Key, 'Result when 'Key : equality>
        (left: 'T[]) 
        (right: 'U[]) 
        (leftKeySelector: 'T -> 'Key) 
        (rightKeySelector: 'U -> 'Key) 
        (resultSelector: 'T -> 'U -> 'Result) : 'Result[] =
        
        [| for l in left do
            for r in right do
                if (leftKeySelector l) = (rightKeySelector r) then
                    yield resultSelector l r |]
    
    /// Group data by a key selector
    let groupData<'T, 'Key when 'Key : equality> (data: 'T[]) (keySelector: 'T -> 'Key) : ('Key * 'T[])[] =
        data
        |> Array.groupBy keySelector
        |> Array.map (fun (key, values) -> key, values)