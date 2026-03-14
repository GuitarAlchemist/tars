namespace Tars.Core

open System.IO
open System.Text.Json

/// ARC-AGI Grid Types and Operations
/// Based on https://github.com/fchollet/ARC-AGI
module ArcTypes =

    /// A grid cell value (0-9, visualized as colors)
    type CellValue = int
    
    /// An ARC grid (2D rectangular matrix of values 0-9)
    type Grid = CellValue array array
    
    /// An input-output example pair
    type ArcPair = {
        Input: Grid
        Output: Grid
    }
    
    /// An ARC task with training examples and test cases
    type ArcTask = {
        Id: string
        TrainingPairs: ArcPair list
        TestPairs: ArcPair list  // Output may be unknown for evaluation
    }
    
    /// Result of applying a transformation
    type TransformResult =
        | Success of Grid
        | Failure of string
    
    /// A transformation hypothesis
    type Transformation =
        | Identity
        | FlipHorizontal
        | FlipVertical
        | Rotate90
        | Rotate180
        | Rotate270
        | Transpose
        | InvertColors  // 0 -> 9, 1 -> 8, etc.
        | FillColor of int
        | ReplaceColor of int * int  // from -> to
        | ExtractSubgrid of int * int * int * int  // x, y, w, h
        | Scale of int  // Scale up by factor
        | Custom of string  // LLM-proposed transformation
    
    // =========================================================================
    // Grid Operations
    // =========================================================================
    
    /// Get grid dimensions
    let dimensions (grid: Grid) : int * int =
        if grid.Length = 0 then (0, 0)
        else (grid.[0].Length, grid.Length)  // (width, height)
    
    /// Check if two grids are equal
    let gridsEqual (g1: Grid) (g2: Grid) : bool =
        let (w1, h1) = dimensions g1
        let (w2, h2) = dimensions g2
        if w1 <> w2 || h1 <> h2 then false
        else
            g1 |> Array.forall2 (fun row1 row2 ->
                row1 |> Array.forall2 (=) row2) g2
    
    /// Convert grid to string representation
    let gridToString (grid: Grid) : string =
        grid
        |> Array.map (fun row -> row |> Array.map string |> String.concat " ")
        |> String.concat "\n"
    
    /// Parse grid from JSON element
    let parseGrid (elem: JsonElement) : Grid =
        elem.EnumerateArray()
        |> Seq.map (fun row ->
            row.EnumerateArray()
            |> Seq.map (fun cell -> cell.GetInt32())
            |> Seq.toArray)
        |> Seq.toArray
    
    /// Parse an ARC task from JSON
    let parseTask (id: string) (json: string) : FSharp.Core.Result<ArcTask, string> =
        try
            let doc = JsonDocument.Parse(json)
            let root = doc.RootElement
            
            let parsePair (elem: JsonElement) : ArcPair =
                { Input = parseGrid (elem.GetProperty("input"))
                  Output = parseGrid (elem.GetProperty("output")) }
            
            let trainPairs =
                root.GetProperty("train").EnumerateArray()
                |> Seq.map parsePair
                |> Seq.toList
            
            let testPairs =
                root.GetProperty("test").EnumerateArray()
                |> Seq.map parsePair
                |> Seq.toList
            
            FSharp.Core.Result.Ok { Id = id; TrainingPairs = trainPairs; TestPairs = testPairs }
        with ex ->
            FSharp.Core.Result.Error $"Failed to parse ARC task: {ex.Message}"
    
    /// Load an ARC task from file
    let loadTask (filePath: string) : FSharp.Core.Result<ArcTask, string> =
        try
            let id = Path.GetFileNameWithoutExtension(filePath)
            let json = File.ReadAllText(filePath)
            parseTask id json
        with ex ->
            FSharp.Core.Result.Error $"Failed to load ARC task file: {ex.Message}"
    
    // =========================================================================
    // Grid Transformations
    // =========================================================================
    
    /// Flip grid horizontally (left-right)
    let flipHorizontal (grid: Grid) : Grid =
        grid |> Array.map Array.rev
    
    /// Flip grid vertically (top-bottom)
    let flipVertical (grid: Grid) : Grid =
        grid |> Array.rev
    
    /// Rotate grid 90 degrees clockwise
    let rotate90 (grid: Grid) : Grid =
        let (w, h) = dimensions grid
        if h = 0 || w = 0 then grid
        else
            Array.init w (fun i ->
                Array.init h (fun j -> grid.[h - 1 - j].[i]))
    
    /// Rotate grid 180 degrees
    let rotate180 (grid: Grid) : Grid =
        grid |> flipVertical |> flipHorizontal
    
    /// Rotate grid 270 degrees clockwise (90 counter-clockwise)
    let rotate270 (grid: Grid) : Grid =
        grid |> rotate90 |> rotate90 |> rotate90
    
    /// Transpose grid (swap rows and columns)
    let transpose (grid: Grid) : Grid =
        let (w, h) = dimensions grid
        if h = 0 || w = 0 then grid
        else
            Array.init w (fun i ->
                Array.init h (fun j -> grid.[j].[i]))
    
    /// Invert colors (0 <-> 9, 1 <-> 8, etc.)
    let invertColors (grid: Grid) : Grid =
        grid |> Array.map (Array.map (fun c -> 9 - c))
    
    /// Fill entire grid with a single color
    let fillColor (color: int) (grid: Grid) : Grid =
        let (w, h) = dimensions grid
        Array.init h (fun _ -> Array.init w (fun _ -> color))
    
    /// Replace one color with another
    let replaceColor (fromColor: int) (toColor: int) (grid: Grid) : Grid =
        grid |> Array.map (Array.map (fun c -> if c = fromColor then toColor else c))
    
    /// Extract a subgrid
    let extractSubgrid (x: int) (y: int) (w: int) (h: int) (grid: Grid) : Grid option =
        let (gw, gh) = dimensions grid
        if x < 0 || y < 0 || x + w > gw || y + h > gh then None
        else
            Some (Array.init h (fun j ->
                Array.init w (fun i -> grid.[y + j].[x + i])))
    
    /// Scale grid up by factor
    let scale (factor: int) (grid: Grid) : Grid =
        grid
        |> Array.collect (fun row ->
            Array.init factor (fun _ ->
                row |> Array.collect (fun cell ->
                    Array.init factor (fun _ -> cell))))
    
    /// Apply a transformation to a grid
    let applyTransform (transform: Transformation) (grid: Grid) : TransformResult =
        try
            match transform with
            | Identity -> Success grid
            | FlipHorizontal -> Success (flipHorizontal grid)
            | FlipVertical -> Success (flipVertical grid)
            | Rotate90 -> Success (rotate90 grid)
            | Rotate180 -> Success (rotate180 grid)
            | Rotate270 -> Success (rotate270 grid)
            | Transpose -> Success (transpose grid)
            | InvertColors -> Success (invertColors grid)
            | FillColor c -> Success (fillColor c grid)
            | ReplaceColor (f, t) -> Success (replaceColor f t grid)
            | ExtractSubgrid (x, y, w, h) ->
                match extractSubgrid x y w h grid with
                | Some g -> Success g
                | None -> Failure "Subgrid extraction failed: out of bounds"
            | Scale f -> Success (scale f grid)
            | Custom name -> Failure $"Custom transformation '{name}' not implemented"
        with ex ->
            Failure $"Transform failed: {ex.Message}"
    
    /// Test if a transformation works for all training examples
    let validateTransform (transform: Transformation) (task: ArcTask) : bool =
        task.TrainingPairs
        |> List.forall (fun pair ->
            match applyTransform transform pair.Input with
            | Success result -> gridsEqual result pair.Output
            | Failure _ -> false)
    
    // =========================================================================
    // Symbolic Description
    // =========================================================================
    
    /// Describe grid properties symbolically
    let describeGrid (grid: Grid) : string list =
        let (w, h) = dimensions grid
        let totalCells = w * h
        
        // Count colors
        let colorCounts =
            grid
            |> Array.collect id
            |> Array.countBy id
            |> Array.sortByDescending snd
        
        let dominantColor = if colorCounts.Length > 0 then Some (fst colorCounts.[0]) else None
        let uniqueColors = colorCounts.Length
        
        // Check symmetry
        let isHSymmetric = gridsEqual grid (flipHorizontal grid)
        let isVSymmetric = gridsEqual grid (flipVertical grid)
        
        // Build description
        [
            $"dimensions: {w}x{h}"
            $"total_cells: {totalCells}"
            $"unique_colors: {uniqueColors}"
            if dominantColor.IsSome then
                $"dominant_color: {dominantColor.Value}"
            yield! colorCounts |> Array.map (fun (c, n) -> $"color_{c}_count: {n}")
            if isHSymmetric then "symmetric_horizontal: true"
            if isVSymmetric then "symmetric_vertical: true"
        ]
    
    /// Describe transformation between two grids
    let describeTransformation (input: Grid) (output: Grid) : string list =
        let (iw, ih) = dimensions input
        let (ow, oh) = dimensions output
        
        [
            if iw = ow && ih = oh then "same_dimensions: true"
            else $"dimension_change: {iw}x{ih} -> {ow}x{oh}"
            
            if gridsEqual output (flipHorizontal input) then "transform: flip_horizontal"
            if gridsEqual output (flipVertical input) then "transform: flip_vertical"
            if gridsEqual output (rotate90 input) then "transform: rotate_90"
            if gridsEqual output (rotate180 input) then "transform: rotate_180"
            if gridsEqual output (rotate270 input) then "transform: rotate_270"
            if gridsEqual output (transpose input) then "transform: transpose"
            if gridsEqual output (invertColors input) then "transform: invert_colors"
        ]
