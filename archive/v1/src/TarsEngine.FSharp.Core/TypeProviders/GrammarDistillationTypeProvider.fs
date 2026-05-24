namespace TarsEngine.FSharp.Core.TypeProviders

open System
open System.IO
open System.Reflection
open Microsoft.FSharp.Core.CompilerServices
open Microsoft.FSharp.Quotations
open ProviderImplementation.ProvidedTypes
open System.Text.Json
open System.Collections.Concurrent

/// Grammar Distillation Type Provider
/// Automatically generates computational expressions from grammar tier definitions
[<TypeProvider>]
type GrammarDistillationTypeProvider(config: TypeProviderConfig) as this =
    inherit TypeProviderForNamespaces(config, assemblyReplacementMap=[("TarsEngine.FSharp.Core.TypeProviders.DesignTime", "TarsEngine.FSharp.Core.TypeProviders")])

    let ns = "TarsEngine.Generated"
    let asm = Assembly.GetExecutingAssembly()
    
    // Grammar tier validation and tracking
    let mutable grammarTiers = ConcurrentDictionary<int, GrammarTier>()
    let mutable lastGrammarUpdate = DateTime.MinValue
    let mutable fileWatcher: FileSystemWatcher option = None
    
    // Grammar tier definition
    type GrammarTier = {
        Tier: int
        Name: string
        Operations: string list
        Dependencies: int list
        Constructs: Map<string, string>
        ComputationalExpressions: string list
        IntegrityHash: string
        CreatedAt: DateTime
        IsValid: bool
    }
    
    // Grammar integrity checker
    module GrammarIntegrity =
        
        let computeHash (tier: GrammarTier) : string =
            let content = sprintf "%d|%s|%s|%s" 
                tier.Tier 
                tier.Name 
                (String.concat "," tier.Operations)
                (String.concat "," tier.ComputationalExpressions)
            System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(content))
            |> Convert.ToHexString
        
        let validateTierDependencies (tier: GrammarTier) (existingTiers: Map<int, GrammarTier>) : bool =
            tier.Dependencies |> List.forall (fun depTier -> 
                existingTiers.ContainsKey(depTier) && existingTiers.[depTier].IsValid)
        
        let validateTierProgression (tier: GrammarTier) (existingTiers: Map<int, GrammarTier>) : bool =
            // Ensure tier progression is logical (each tier builds on previous)
            if tier.Tier = 1 then true
            else
                existingTiers.ContainsKey(tier.Tier - 1) && 
                tier.Operations.Length >= existingTiers.[tier.Tier - 1].Operations.Length
        
        let validateGrammarIntegrity (tier: GrammarTier) (existingTiers: Map<int, GrammarTier>) : bool =
            let hashValid = computeHash tier = tier.IntegrityHash
            let dependenciesValid = validateTierDependencies tier existingTiers
            let progressionValid = validateTierProgression tier existingTiers
            
            hashValid && dependenciesValid && progressionValid
    
    // Computational expression generator
    module ComputationalExpressionGenerator =
        
        let generateBuilderType (tier: GrammarTier) : ProvidedTypeDefinition =
            let builderName = sprintf "%sBuilder" tier.Name
            let builderType = ProvidedTypeDefinition(builderName, Some typeof<obj>, isErased = false)
            
            // Add Return method
            let returnMethod = ProvidedMethod("Return", 
                [ProvidedParameter("value", typeof<obj>)], 
                typeof<obj>,
                fun args -> <@@ %%args.[1] : obj @@>)
            builderType.AddMember(returnMethod)
            
            // Add Bind method
            let bindMethod = ProvidedMethod("Bind",
                [ProvidedParameter("value", typeof<obj>); ProvidedParameter("f", typeof<obj -> obj>)],
                typeof<obj>,
                fun args -> <@@ (%%args.[2] : obj -> obj) (%%args.[1] : obj) @@>)
            builderType.AddMember(bindMethod)
            
            // Add Zero method
            let zeroMethod = ProvidedMethod("Zero", [], typeof<obj>,
                fun args -> <@@ null : obj @@>)
            builderType.AddMember(zeroMethod)
            
            // Add Combine method
            let combineMethod = ProvidedMethod("Combine",
                [ProvidedParameter("a", typeof<obj>); ProvidedParameter("b", typeof<obj>)],
                typeof<obj>,
                fun args -> <@@ (%%args.[1] : obj, %%args.[2] : obj) @@>)
            builderType.AddMember(combineMethod)
            
            builderType
        
        let generateOperationType (operation: string) (tier: GrammarTier) : ProvidedTypeDefinition =
            let operationType = ProvidedTypeDefinition(operation, Some typeof<obj>, isErased = false)
            
            // Add operation-specific methods based on tier
            match tier.Tier with
            | 1 -> // Basic constructs
                let executeMethod = ProvidedMethod("Execute", [], typeof<obj>,
                    fun args -> <@@ sprintf "Executing %s (Tier 1)" operation @@>)
                operationType.AddMember(executeMethod)
            
            | 2 -> // Computational expressions
                let computeMethod = ProvidedMethod("Compute", 
                    [ProvidedParameter("input", typeof<obj>)], typeof<obj>,
                    fun args -> <@@ sprintf "Computing %s with %A (Tier 2)" operation %%args.[1] @@>)
                operationType.AddMember(computeMethod)
            
            | 3 -> // Advanced operations
                let advancedMethod = ProvidedMethod("AdvancedOperation",
                    [ProvidedParameter("input", typeof<obj>); ProvidedParameter("params", typeof<obj array>)], typeof<obj>,
                    fun args -> <@@ sprintf "Advanced %s operation with %A and params %A (Tier 3)" operation %%args.[1] %%args.[2] @@>)
                operationType.AddMember(advancedMethod)
            
            | 4 -> // Domain-specific operations
                let domainMethod = ProvidedMethod("DomainSpecificOperation",
                    [ProvidedParameter("context", typeof<obj>); ProvidedParameter("data", typeof<obj>)], typeof<obj>,
                    fun args -> <@@ sprintf "Domain-specific %s operation in context %A with data %A (Tier 4)" operation %%args.[1] %%args.[2] @@>)
                operationType.AddMember(domainMethod)
            
            | _ -> // Future tiers
                let futureMethod = ProvidedMethod("FutureOperation", [], typeof<obj>,
                    fun args -> <@@ sprintf "Future tier %d operation: %s" tier.Tier operation @@>)
                operationType.AddMember(futureMethod)
            
            operationType
        
        let generateTierModule (tier: GrammarTier) : ProvidedTypeDefinition =
            let moduleName = sprintf "Tier%d_%s" tier.Tier tier.Name
            let moduleType = ProvidedTypeDefinition(moduleName, Some typeof<obj>, isErased = false)
            
            // Generate builder type
            let builderType = generateBuilderType tier
            moduleType.AddMember(builderType)
            
            // Generate operation types
            for operation in tier.Operations do
                let operationType = generateOperationType operation tier
                moduleType.AddMember(operationType)
            
            // Add tier metadata
            let tierProperty = ProvidedProperty("TierLevel", typeof<int>,
                getterCode = fun args -> <@@ tier.Tier @@>)
            moduleType.AddMember(tierProperty)
            
            let nameProperty = ProvidedProperty("TierName", typeof<string>,
                getterCode = fun args -> <@@ tier.Name @@>)
            moduleType.AddMember(nameProperty)
            
            let operationsProperty = ProvidedProperty("Operations", typeof<string array>,
                getterCode = fun args -> <@@ tier.Operations |> List.toArray @@>)
            moduleType.AddMember(operationsProperty)
            
            moduleType
    
    // Grammar file watcher
    module GrammarWatcher =
        
        let parseGrammarFile (filePath: string) : GrammarTier option =
            try
                if File.Exists(filePath) then
                    let content = File.ReadAllText(filePath)
                    let json = JsonDocument.Parse(content)
                    let root = json.RootElement
                    
                    let tier = {
                        Tier = root.GetProperty("tier").GetInt32()
                        Name = root.GetProperty("name").GetString()
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
                    }
                    
                    let tierWithHash = { tier with IntegrityHash = GrammarIntegrity.computeHash tier }
                    Some tierWithHash
                else
                    None
            with
            | ex -> 
                printfn "Error parsing grammar file %s: %s" filePath ex.Message
                None
        
        let onGrammarFileChanged (filePath: string) =
            match parseGrammarFile filePath with
            | Some tier ->
                let existingTiers = grammarTiers |> Seq.map (fun kvp -> kvp.Key, kvp.Value) |> Map.ofSeq
                
                if GrammarIntegrity.validateGrammarIntegrity tier existingTiers then
                    let validTier = { tier with IsValid = true }
                    grammarTiers.TryAdd(tier.Tier, validTier) |> ignore
                    lastGrammarUpdate <- DateTime.UtcNow
                    
                    printfn "✅ Grammar Tier %d (%s) validated and loaded" tier.Tier tier.Name
                    printfn "   Operations: %s" (String.concat ", " tier.Operations)
                    printfn "   Computational Expressions: %d" tier.ComputationalExpressions.Length
                    
                    // Trigger type provider refresh
                    this.Invalidate()
                else
                    printfn "❌ Grammar Tier %d (%s) failed integrity validation" tier.Tier tier.Name
            | None ->
                printfn "❌ Failed to parse grammar file: %s" filePath
        
        let startWatching (grammarDirectory: string) =
            if Directory.Exists(grammarDirectory) then
                let watcher = new FileSystemWatcher(grammarDirectory, "*.json")
                watcher.NotifyFilter <- NotifyFilters.LastWrite ||| NotifyFilters.FileName
                watcher.Changed.Add(fun e -> onGrammarFileChanged e.FullPath)
                watcher.Created.Add(fun e -> onGrammarFileChanged e.FullPath)
                watcher.EnableRaisingEvents <- true
                
                fileWatcher <- Some watcher
                printfn "🔍 Grammar file watcher started for: %s" grammarDirectory
                
                // Load existing grammar files
                Directory.GetFiles(grammarDirectory, "*.json")
                |> Array.iter onGrammarFileChanged
            else
                printfn "⚠️ Grammar directory not found: %s" grammarDirectory
    
    // Initialize the type provider
    do
        let grammarDirectory = Path.Combine(config.ResolutionFolder, ".tars", "grammars")
        GrammarWatcher.startWatching grammarDirectory
        
        // Create the main provided type
        let providedType = ProvidedTypeDefinition(asm, ns, "GrammarDistillation", Some typeof<obj>, isErased = false)
        
        // Add dynamic type generation based on grammar tiers
        providedType.AddMembersDelayed(fun () ->
            let currentTiers = grammarTiers |> Seq.map (fun kvp -> kvp.Value) |> Seq.filter (fun t -> t.IsValid) |> Seq.toList
            
            let generatedTypes = 
                currentTiers
                |> List.map ComputationalExpressionGenerator.generateTierModule
            
            // Add metadata type
            let metadataType = ProvidedTypeDefinition("GrammarMetadata", Some typeof<obj>, isErased = false)
            
            let tierCountProperty = ProvidedProperty("TierCount", typeof<int>,
                getterCode = fun args -> <@@ currentTiers.Length @@>)
            metadataType.AddMember(tierCountProperty)
            
            let lastUpdateProperty = ProvidedProperty("LastUpdate", typeof<DateTime>,
                getterCode = fun args -> <@@ lastGrammarUpdate @@>)
            metadataType.AddMember(lastUpdateProperty)
            
            let tierNamesProperty = ProvidedProperty("TierNames", typeof<string array>,
                getterCode = fun args -> <@@ currentTiers |> List.map (fun t -> t.Name) |> List.toArray @@>)
            metadataType.AddMember(tierNamesProperty)
            
            metadataType :: generatedTypes
        )
        
        this.AddNamespace(ns, [providedType])
    
    // Cleanup
    interface IDisposable with
        member _.Dispose() =
            match fileWatcher with
            | Some watcher -> 
                watcher.Dispose()
                fileWatcher <- None
            | None -> ()

[<assembly: TypeProviderAssembly>]
do ()
