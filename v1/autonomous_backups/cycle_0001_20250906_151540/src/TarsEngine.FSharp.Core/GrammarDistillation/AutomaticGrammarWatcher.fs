namespace TarsEngine.FSharp.Core.GrammarDistillation

open System
open System.IO
open System.Text.Json
open System.Collections.Concurrent
open System.Security.Cryptography
open System.Text

/// Automatic Grammar Watcher and Computational Expression Generator
/// Monitors grammar tier files and generates F# computational expressions on-the-fly
module AutomaticGrammarWatcher =

    // ============================================================================
    // GRAMMAR TIER DEFINITIONS
    // ============================================================================

    type GrammarTier = {
        Tier: int
        Name: string
        Description: string
        Operations: string list
        Dependencies: int list
        Constructs: Map<string, string>
        ComputationalExpressions: string list
        IntegrityHash: string
        CreatedAt: DateTime
        IsValid: bool
        ValidationErrors: string list
    }

    type GrammarEvent =
        | TierAdded of GrammarTier
        | TierUpdated of GrammarTier * GrammarTier  // old, new
        | TierRemoved of int
        | TierValidationFailed of int * string list
        | IntegrityCheckPassed of int
        | IntegrityCheckFailed of int * string

    // ============================================================================
    // GRAMMAR INTEGRITY VALIDATION
    // ============================================================================

    module GrammarIntegrity =
        
        let computeHash (tier: GrammarTier) : string =
            let content = sprintf "%d|%s|%s|%s|%s" 
                tier.Tier 
                tier.Name 
                tier.Description
                (String.concat "," tier.Operations)
                (String.concat "," tier.ComputationalExpressions)
            
            use sha256 = SHA256.Create()
            let hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(content))
            Convert.ToHexString(hashBytes)
        
        let validateTierStructure (tier: GrammarTier) : string list =
            let errors = ResizeArray<string>()
            
            // Validate tier number
            if tier.Tier < 1 || tier.Tier > 10 then
                errors.Add(sprintf "Invalid tier number: %d (must be 1-10)" tier.Tier)
            
            // Validate name
            if String.IsNullOrWhiteSpace(tier.Name) then
                errors.Add("Tier name cannot be empty")
            
            // Validate operations
            if tier.Operations.IsEmpty then
                errors.Add("Tier must have at least one operation")
            
            // Validate computational expressions for tier 2+
            if tier.Tier >= 2 && tier.ComputationalExpressions.IsEmpty then
                errors.Add("Tier 2+ must have computational expressions")
            
            errors |> Seq.toList
        
        let validateTierDependencies (tier: GrammarTier) (existingTiers: Map<int, GrammarTier>) : string list =
            let errors = ResizeArray<string>()

            for depTier in tier.Dependencies do
                if not (existingTiers.ContainsKey(depTier)) then
                    errors.Add(sprintf "Missing dependency: Tier %d" depTier)
                elif not existingTiers.[depTier].IsValid then
                    errors.Add(sprintf "Invalid dependency: Tier %d is not valid" depTier)

            errors |> Seq.toList
        
        let validateTierProgression (tier: GrammarTier) (existingTiers: Map<int, GrammarTier>) : string list =
            let errors = ResizeArray<string>()
            
            // Tier 1 has no prerequisites
            if tier.Tier = 1 then
                []
            else
                // Check if previous tier exists
                if not (existingTiers.ContainsKey(tier.Tier - 1)) then
                    errors.Add(sprintf "Missing prerequisite: Tier %d requires Tier %d" tier.Tier (tier.Tier - 1))
                else
                    let prevTier = existingTiers.[tier.Tier - 1]
                    
                    // Check operation count progression
                    if tier.Operations.Length < prevTier.Operations.Length then
                        errors.Add(sprintf "Operation regression: Tier %d has fewer operations than Tier %d" tier.Tier (tier.Tier - 1))
                    
                    // Check computational expression progression
                    if tier.Tier >= 2 && tier.ComputationalExpressions.Length < prevTier.ComputationalExpressions.Length then
                        errors.Add(sprintf "Expression regression: Tier %d has fewer expressions than Tier %d" tier.Tier (tier.Tier - 1))
            
            errors |> Seq.toList
        
        let validateGrammarIntegrity (tier: GrammarTier) (existingTiers: Map<int, GrammarTier>) : bool * string list =
            let allErrors = ResizeArray<string>()
            
            // Validate structure
            allErrors.AddRange(validateTierStructure tier)
            
            // Validate dependencies
            allErrors.AddRange(validateTierDependencies tier existingTiers)
            
            // Validate progression
            allErrors.AddRange(validateTierProgression tier existingTiers)
            
            // Validate hash integrity
            let expectedHash = computeHash tier
            if tier.IntegrityHash <> expectedHash then
                allErrors.Add(sprintf "Hash mismatch: expected %s, got %s" expectedHash tier.IntegrityHash)
            
            let errors = allErrors |> Seq.toList
            (errors.IsEmpty, errors)

    // ============================================================================
    // COMPUTATIONAL EXPRESSION GENERATOR
    // ============================================================================

    module ComputationalExpressionGenerator =
        
        let generateBuilderCode (tier: GrammarTier) : string =
            let builderName = sprintf "%sBuilder" tier.Name
            
            sprintf """
/// Generated computational expression builder for %s (Tier %d)
type %s() =
    member _.Return(value: 'T) = value
    member _.ReturnFrom(value: 'T) = value
    member _.Bind(value: 'T, f: 'T -> 'U) = f value
    member _.Zero() = Unchecked.defaultof<'T>
    member _.Combine(a: 'T, b: 'T) = 
        // Tier %d combination logic
        match box a, box b with
        | :? string as sa, :? string as sb -> box (sa + sb) :?> 'T
        | :? int as ia, :? int as ib -> box (ia + ib) :?> 'T
        | :? float as fa, :? float as fb -> box (fa + fb) :?> 'T
        | _ -> a
    
    member _.Delay(f: unit -> 'T) = f()
    
    member _.For(sequence: seq<'U>, body: 'U -> 'T) =
        sequence |> Seq.fold (fun acc item -> 
            let result = body item
            // Combine results based on tier level
            acc
        ) (Unchecked.defaultof<'T>)

let %s = %s()
""" tier.Description tier.Tier builderName tier.Tier (tier.Name.ToLower()) builderName
        
        let generateOperationCode (operation: string) (tier: GrammarTier) : string =
            match tier.Tier with
            | 1 -> // Basic constructs
                sprintf """
/// %s operation (Tier 1: Basic Construct)
let %s (input: 'T) : 'T =
    // Basic tier 1 operation
    input
""" operation operation
            
            | 2 -> // Computational expressions
                sprintf """
/// %s operation (Tier 2: Computational Expression)
let %s (input: 'T) : 'T =
    %s {
        let! value = input
        return value
    }
""" operation operation (tier.Name.ToLower())
            
            | 3 -> // Advanced operations
                sprintf """
/// %s operation (Tier 3: Advanced Operation)
let %s (input: 'T) (parameters: 'U array) : 'T =
    %s {
        let! value = input
        // Advanced processing with parameters
        return value
    }
""" operation operation (tier.Name.ToLower())
            
            | 4 -> // Domain-specific operations
                sprintf """
/// %s operation (Tier 4: Domain-Specific)
let %s (context: 'Context) (input: 'T) : 'T =
    %s {
        let! value = input
        // Domain-specific processing in context
        return value
    }
""" operation operation (tier.Name.ToLower())
            
            | _ -> // Future tiers
                sprintf """
/// %s operation (Tier %d: Future Extension)
let %s (input: 'T) : 'T =
    // Future tier operation
    input
""" operation tier.Tier operation
        
        let generateTierModule (tier: GrammarTier) : string =
            let moduleHeader = sprintf """
/// Generated module for %s (Tier %d)
/// %s
/// Generated at: %s
/// Operations: %d
/// Computational Expressions: %d
module %s =
""" tier.Name tier.Tier tier.Description (tier.CreatedAt.ToString("yyyy-MM-dd HH:mm:ss")) tier.Operations.Length tier.ComputationalExpressions.Length tier.Name
            
            let builderCode = generateBuilderCode tier
            
            let operationCodes = 
                tier.Operations 
                |> List.map (fun op -> generateOperationCode op tier)
                |> String.concat "\n"
            
            let metadataCode = sprintf """
    /// Tier metadata
    let tierLevel = %d
    let tierName = "%s"
    let operations = [| %s |]
    let computationalExpressions = [| %s |]
    let createdAt = System.DateTime.Parse("%s")
    let isValid = %b
""" tier.Tier tier.Name 
    (tier.Operations |> List.map (sprintf "\"%s\"") |> String.concat "; ")
    (tier.ComputationalExpressions |> List.map (sprintf "\"%s\"") |> String.concat "; ")
    (tier.CreatedAt.ToString("O"))
    tier.IsValid
            
            moduleHeader + builderCode + operationCodes + metadataCode
        
        let generateCompleteFile (tiers: GrammarTier list) : string =
            let header = sprintf """
// Auto-generated F# computational expressions from grammar tiers
// Generated at: %s
// Total tiers: %d
// Valid tiers: %d

namespace TarsEngine.Generated.GrammarDistillation

open System

""" (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")) tiers.Length (tiers |> List.filter (fun t -> t.IsValid) |> List.length)
            
            let tierModules = 
                tiers 
                |> List.filter (fun t -> t.IsValid)
                |> List.sortBy (fun t -> t.Tier)
                |> List.map generateTierModule
                |> String.concat "\n"
            
            header + tierModules

    // ============================================================================
    // GRAMMAR FILE WATCHER
    // ============================================================================

    type GrammarWatcher(grammarDirectory: string) =
        let grammarTiers = ConcurrentDictionary<int, GrammarTier>()
        let mutable fileWatcher: FileSystemWatcher option = None
        let grammarEvents = Event<GrammarEvent>()
        let mutable lastUpdate = DateTime.MinValue
        
        [<CLIEvent>]
        member _.GrammarEvents = grammarEvents.Publish
        
        member _.CurrentTiers = 
            grammarTiers 
            |> Seq.map (fun kvp -> kvp.Value) 
            |> Seq.filter (fun t -> t.IsValid)
            |> Seq.sortBy (fun t -> t.Tier)
            |> Seq.toList
        
        member _.LastUpdate = lastUpdate
        
        member private _.ParseGrammarFile(filePath: string) : GrammarTier option =
            try
                if File.Exists(filePath) then
                    let content = File.ReadAllText(filePath)
                    let json = JsonDocument.Parse(content)
                    let root = json.RootElement
                    
                    let tier = {
                        Tier = root.GetProperty("tier").GetInt32()
                        Name = root.GetProperty("name").GetString()
                        Description = 
                            if root.TryGetProperty("description", &Unchecked.defaultof<JsonElement>) then
                                root.GetProperty("description").GetString()
                            else ""
                        Operations = 
                            root.GetProperty("operations").EnumerateArray()
                            |> Seq.map (fun x -> x.GetString())
                            |> Seq.toList
                        Dependencies = 
                            if root.TryGetProperty("dependencies", &Unchecked.defaultof<JsonElement>) then
                                root.GetProperty("dependencies").EnumerateArray()
                                |> Seq.map (fun x -> x.GetInt32())
                                |> Seq.toList
                            else []
                        Constructs = 
                            if root.TryGetProperty("constructs", &Unchecked.defaultof<JsonElement>) then
                                root.GetProperty("constructs").EnumerateObject()
                                |> Seq.map (fun prop -> prop.Name, prop.Value.GetString())
                                |> Map.ofSeq
                            else Map.empty
                        ComputationalExpressions =
                            if root.TryGetProperty("computationalExpressions", &Unchecked.defaultof<JsonElement>) then
                                root.GetProperty("computationalExpressions").EnumerateArray()
                                |> Seq.map (fun x -> x.GetString())
                                |> Seq.toList
                            else []
                        IntegrityHash = ""
                        CreatedAt = DateTime.UtcNow
                        IsValid = false
                        ValidationErrors = []
                    }
                    
                    let tierWithHash = { tier with IntegrityHash = GrammarIntegrity.computeHash tier }
                    Some tierWithHash
                else
                    None
            with
            | ex -> 
                printfn "❌ Error parsing grammar file %s: %s" filePath ex.Message
                None
        
        member private this.OnGrammarFileChanged(filePath: string) =
            match this.ParseGrammarFile(filePath) with
            | Some tier ->
                let existingTiers = grammarTiers |> Seq.map (fun kvp -> kvp.Key, kvp.Value) |> Map.ofSeq
                let (isValid, errors) = GrammarIntegrity.validateGrammarIntegrity tier existingTiers
                
                if isValid then
                    let validTier = { tier with IsValid = true; ValidationErrors = [] }
                    let wasUpdate = grammarTiers.ContainsKey(tier.Tier)
                    
                    if wasUpdate then
                        let oldTier = grammarTiers.[tier.Tier]
                        grammarTiers.TryUpdate(tier.Tier, validTier, oldTier) |> ignore
                        grammarEvents.Trigger(TierUpdated(oldTier, validTier))
                    else
                        grammarTiers.TryAdd(tier.Tier, validTier) |> ignore
                        grammarEvents.Trigger(TierAdded(validTier))
                    
                    lastUpdate <- DateTime.UtcNow
                    grammarEvents.Trigger(IntegrityCheckPassed(tier.Tier))
                    
                    printfn "✅ Grammar Tier %d (%s) validated and loaded" tier.Tier tier.Name
                    printfn "   Operations: %s" (String.concat ", " tier.Operations)
                    printfn "   Computational Expressions: %d" tier.ComputationalExpressions.Length
                    
                    // Generate computational expressions
                    this.GenerateComputationalExpressions()
                else
                    let invalidTier = { tier with IsValid = false; ValidationErrors = errors }
                    grammarEvents.Trigger(TierValidationFailed(tier.Tier, errors))
                    grammarEvents.Trigger(IntegrityCheckFailed(tier.Tier, String.concat "; " errors))
                    
                    printfn "❌ Grammar Tier %d (%s) failed integrity validation:" tier.Tier tier.Name
                    for error in errors do
                        printfn "   - %s" error
            | None ->
                printfn "❌ Failed to parse grammar file: %s" filePath
        
        member private this.GenerateComputationalExpressions() =
            let validTiers = this.CurrentTiers
            if not validTiers.IsEmpty then
                let generatedCode = ComputationalExpressionGenerator.generateCompleteFile validTiers
                let outputPath = Path.Combine(grammarDirectory, "..", "Generated", "GrammarDistillationExpressions.fs")
                
                // Ensure output directory exists
                let outputDir = Path.GetDirectoryName(outputPath)
                if not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                
                File.WriteAllText(outputPath, generatedCode)
                printfn "🎨 Generated computational expressions: %s" outputPath
                printfn "   Tiers: %s" (validTiers |> List.map (fun t -> sprintf "%d(%s)" t.Tier t.Name) |> String.concat ", ")
        
        member this.StartWatching() =
            if Directory.Exists(grammarDirectory) then
                let watcher = new FileSystemWatcher(grammarDirectory, "*.json")
                watcher.NotifyFilter <- NotifyFilters.LastWrite ||| NotifyFilters.FileName ||| NotifyFilters.CreationTime
                watcher.Changed.Add(fun e -> this.OnGrammarFileChanged(e.FullPath))
                watcher.Created.Add(fun e -> this.OnGrammarFileChanged(e.FullPath))
                watcher.Deleted.Add(fun e -> 
                    let fileName = Path.GetFileNameWithoutExtension(e.Name)
                    if Int32.TryParse(fileName, &Unchecked.defaultof<int>) then
                        let tierNum = Int32.Parse(fileName)
                        if grammarTiers.TryRemove(tierNum, &Unchecked.defaultof<GrammarTier>) then
                            grammarEvents.Trigger(TierRemoved(tierNum))
                            printfn "🗑️ Grammar Tier %d removed" tierNum
                )
                watcher.EnableRaisingEvents <- true
                
                fileWatcher <- Some watcher
                printfn "🔍 Grammar file watcher started for: %s" grammarDirectory
                
                // Load existing grammar files
                Directory.GetFiles(grammarDirectory, "*.json")
                |> Array.iter this.OnGrammarFileChanged
            else
                printfn "⚠️ Grammar directory not found: %s" grammarDirectory
        
        member _.StopWatching() =
            match fileWatcher with
            | Some watcher -> 
                watcher.Dispose()
                fileWatcher <- None
                printfn "🛑 Grammar file watcher stopped"
            | None -> ()
        
        interface IDisposable with
            member this.Dispose() = this.StopWatching()
