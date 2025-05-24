namespace TarsEngine.FSharp.Core.Working

/// <summary>
/// Module containing collection utilities.
/// </summary>
module Collections =
    
    /// <summary>
    /// Safely gets an item from a list by index.
    /// </summary>
    let tryItem index list =
        if index >= 0 && index < List.length list then
            Some (List.item index list)
        else
            None
    
    /// <summary>
    /// Safely gets an item from an array by index.
    /// </summary>
    let tryItemArray index array =
        if index >= 0 && index < Array.length array then
            Some (Array.item index array)
        else
            None
    
    /// <summary>
    /// Chunks a list into sublists of specified size.
    /// </summary>
    let chunk size list =
        let rec chunkHelper acc current currentSize remaining =
            match remaining with
            | [] when List.isEmpty current -> List.rev acc
            | [] -> List.rev (List.rev current :: acc)
            | head :: tail when currentSize < size ->
                chunkHelper acc (head :: current) (currentSize + 1) tail
            | head :: tail ->
                chunkHelper (List.rev current :: acc) [head] 1 tail
        
        chunkHelper [] [] 0 list
    
    /// <summary>
    /// Groups consecutive elements that satisfy a predicate.
    /// </summary>
    let groupConsecutive predicate list =
        let rec groupHelper acc current remaining =
            match remaining with
            | [] when List.isEmpty current -> List.rev acc
            | [] -> List.rev (List.rev current :: acc)
            | head :: tail when predicate head ->
                groupHelper acc (head :: current) tail
            | head :: tail when not (List.isEmpty current) ->
                groupHelper (List.rev current :: acc) [head] tail
            | head :: tail ->
                groupHelper acc [head] tail
        
        groupHelper [] [] list
    
    /// <summary>
    /// Finds the first duplicate in a list.
    /// </summary>
    let findFirstDuplicate list =
        let rec findHelper seen remaining =
            match remaining with
            | [] -> None
            | head :: tail when Set.contains head seen -> Some head
            | head :: tail -> findHelper (Set.add head seen) tail
        
        findHelper Set.empty list
    
    /// <summary>
    /// Removes duplicates from a list while preserving order.
    /// </summary>
    let distinct list =
        let rec distinctHelper acc seen remaining =
            match remaining with
            | [] -> List.rev acc
            | head :: tail when Set.contains head seen ->
                distinctHelper acc seen tail
            | head :: tail ->
                distinctHelper (head :: acc) (Set.add head seen) tail
        
        distinctHelper [] Set.empty list
