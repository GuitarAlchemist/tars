module Tars.Registry.Registry

open System
open System.Reflection
open System.Text.Json
open System.Text.Json.Nodes

/// Build a TarsSkill record from a discovered (attribute, method) pair.
/// The handler invokes the static method via reflection. It accepts a
/// `JsonNode` and tolerates three handler signatures:
///   1. (JsonNode) -> JsonNode
///   2. (JsonNode) -> Result<JsonNode, string>
///   3. (unit)     -> JsonNode  (no-arg skills)
/// Any other shape is rejected at discovery time.
let private buildSkill (attr: TarsSkillAttribute) (m: MethodInfo) : TarsSkill =
    let invoke (input: JsonNode) : Result<JsonNode, string> =
        try
            let args : obj array =
                match m.GetParameters() with
                | [||] -> [||]
                | [| p |] when p.ParameterType = typeof<JsonNode> -> [| box input |]
                | parameters ->
                    // Best-effort: pass the JsonNode for the first parameter,
                    // null for the rest. Surfaces as an exception → Error.
                    parameters
                    |> Array.mapi (fun i _ -> if i = 0 then box input else null)

            let raw = m.Invoke(null, args)
            match raw with
            | :? Result<JsonNode, string> as r -> r
            | :? JsonNode as n -> Ok n
            | null -> Ok (JsonValue.Create(null :> obj) :> JsonNode)
            | other ->
                // Fall back to round-tripping unknown return values through
                // System.Text.Json so handlers can return anonymous records.
                let json = JsonSerializer.Serialize(other)
                Ok (JsonNode.Parse(json))
        with ex ->
            Error ex.Message

    let schema () : JsonNode =
        // No first-class schema source yet — skills can override via a
        // companion "<methodname>Schema" static method discovered later.
        // For now return an empty object so MCP tools/list stays valid.
        JsonObject() :> JsonNode

    {
        Name = attr.Name
        Domain = attr.Domain
        Description = attr.Description
        Schema = schema
        Handler = invoke
    }

/// Reflect over every loaded assembly and collect all methods carrying
/// `[<TarsSkill>]`. ReflectionTypeLoadException is swallowed per-assembly
/// so a single bad plug-in cannot poison the whole registry.
let private discover () : TarsSkill array =
    AppDomain.CurrentDomain.GetAssemblies()
    |> Array.collect (fun a ->
        try
            a.GetTypes()
        with
        | :? ReflectionTypeLoadException as ex ->
            ex.Types |> Array.filter (fun t -> not (isNull t)))
    |> Array.collect (fun t ->
        try
            t.GetMethods(BindingFlags.Public ||| BindingFlags.Static)
        with _ -> [||])
    |> Array.choose (fun m ->
        let attr = m.GetCustomAttribute<TarsSkillAttribute>()
        if isNull attr then None
        else Some (buildSkill attr m))

let private cache = lazy (discover ())

/// All discovered skills, computed once and memoized.
let all () : TarsSkill array = cache.Value

/// Look up a skill by its registered dotted name.
let byName (name: string) : TarsSkill option =
    cache.Value |> Array.tryFind (fun s -> s.Name = name)
