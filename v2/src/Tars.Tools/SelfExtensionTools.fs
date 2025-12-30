namespace Tars.Tools.Standard

open System
open System.IO
open Tars.Tools
open Tars.Core.SelfExtension

module SelfExtensionTools =

    /// Helper to get paths
    let private getPaths () =
        SelfExtensionService.defaultPaths Environment.CurrentDirectory

    [<TarsToolAttribute("create_dynamic_tool",
        "Creates a new dynamic tool for TARS. 
EXECUTION CONTRACT:
1. Implementation must be a standalone F# script (.fsx).
2. Input JSON is passed as the first argument: fsi.CommandLineArgs.[1]
3. The script MUST print its final result to stdout (e.g. using printfn).
4. Do not just define a function; you must call it or process the input and print the result.

Input JSON: { \"name\": \"my_tool\", \"description\": \"...\", \"implementation_code\": \"...F# script...\", \"input_schema\": \"...\", \"output_schema\": \"...\" }")>]
    let createDynamicTool (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                
                let name = root.GetProperty("name").GetString()
                let desc = root.GetProperty("description").GetString()
                let code = root.GetProperty("implementation_code").GetString()
                
                let inputSchema =
                    if root.TryGetProperty("input_schema", ref Unchecked.defaultof<System.Text.Json.JsonElement>) 
                    then Some (root.GetProperty("input_schema").GetString()) else None
                
                let outputSchema =
                    if root.TryGetProperty("output_schema", ref Unchecked.defaultof<System.Text.Json.JsonElement>) 
                    then Some (root.GetProperty("output_schema").GetString()) else None

                let toolDef : DynamicToolDefinition =
                    { Name = name
                      Description = desc
                      Version = "1.0.0"
                      InputSchema = inputSchema
                      OutputSchema = outputSchema
                      Implementation = FSharpScript code }
                
                let ext = SelfExtensionService.generateTool (getPaths()) toolDef
                
                return $"✅ Created dynamic tool '{name}' (ID: {ext.Id}). It will be available in the next execution cycle."
            with ex ->
                return $"❌ Failed to create tool: {ex.Message}"
        }

    [<TarsToolAttribute("create_metascript",
        "Creates a new Metascript workflow. Input JSON: { \"name\": \"my_workflow\", \"description\": \"...\", \"blocks\": [\"block1\", \"block2\"] }")>]
    let createMetascript (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                
                let name = root.GetProperty("name").GetString()
                let desc = root.GetProperty("description").GetString()
                let blocks =
                    root.GetProperty("blocks").EnumerateArray() 
                    |> Seq.map (fun e -> e.GetString()) 
                    |> Seq.toList
                
                let ext = SelfExtensionService.generateMetascript (getPaths()) name desc blocks
                
                return $"✅ Created metascript '{name}' (ID: {ext.Id})."
            with ex ->
                return $"❌ Failed to create metascript: {ex.Message}"
        }

    [<TarsToolAttribute("create_grammar",
        "Creates a new EBNF grammar. Input JSON: { \"name\": \"my_grammar\", \"description\": \"...\", \"ebnf\": \"...\" }")>]
    let createGrammar (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                
                let name = root.GetProperty("name").GetString()
                let desc = root.GetProperty("description").GetString()
                let ebnf = root.GetProperty("ebnf").GetString()
                
                let ext = SelfExtensionService.generateGrammar (getPaths()) name desc ebnf
                
                return $"✅ Created grammar '{name}' (ID: {ext.Id})."
            with ex ->
                return $"❌ Failed to create grammar: {ex.Message}"
        }

    [<TarsToolAttribute("create_block",
        "Creates a new DSL block definition. Input JSON: { \"name\": \"MY_BLOCK\", \"description\": \"...\", \"ebnf_rule\": \"...\", \"parameters\": [{\"name\":\"param1\",\"type\":\"string\",\"required\":true}] }")>]
    let createBlock (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                
                let name = root.GetProperty("name").GetString()
                let desc = root.GetProperty("description").GetString()
                let ebnf = root.GetProperty("ebnf_rule").GetString()
                
                let parameters =
                    if root.TryGetProperty("parameters", ref Unchecked.defaultof<System.Text.Json.JsonElement>) then
                        root.GetProperty("parameters").EnumerateArray()
                        |> Seq.map (fun p -> 
                            { Name = p.GetProperty("name").GetString()
                              Type = p.GetProperty("type").GetString()
                              Required = p.GetProperty("required").GetBoolean()
                              Default = if p.TryGetProperty("default", ref Unchecked.defaultof<System.Text.Json.JsonElement>) then Some(p.GetProperty("default").GetString()) else None })
                        |> Seq.toList
                    else []
                
                let blockDef : BlockDefinition =
                    { Name = name
                      Description = desc
                      Version = "1.0.0"
                      Parameters = parameters
                      ContentType = BlockContentType.Freeform
                      EbnfRule = ebnf
                      CompileToIR = "" }
                
                let ext = SelfExtensionService.generateBlock (getPaths()) blockDef
                
                return $"✅ Created block '{name}' (ID: {ext.Id})."
            with ex ->
                return $"❌ Failed to create block: {ex.Message}"
        }

    [<TarsToolAttribute("list_extensions", "Lists all self-generated extensions (tools, grammars, metascripts). No input required.")>]
    let listExtensions (_: string) =
        task {
            try
                let report = SelfExtensionService.generateReport (getPaths())
                return report
            with ex ->
                return $"❌ Failed to list extensions: {ex.Message}"
        }
