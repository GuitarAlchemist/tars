namespace Tars.Core

open System
open System.Text.Json
open System.Text.Json.Serialization

/// <summary>
/// Handles JSON-LD serialization for Semantic Messages.
/// Ensures compliance with FIPA-ACL and JSON-LD standards.
/// </summary>
module SemanticSerialization =

    /// <summary>
    /// JSON-LD Context definition for TARS messages.
    /// </summary>
    let private jsonLdContext =
        Map
            [ "tars", "http://tars.ai/ns#"
              "fipa", "http://fipa.org/ns#"
              "performative", "fipa:performative"
              "sender", "fipa:sender"
              "receiver", "fipa:receiver"
              "content", "fipa:content"
              "language", "fipa:language"
              "ontology", "fipa:ontology"
              "intent", "tars:intent"
              "constraints", "tars:constraints"
              "correlationId", "tars:correlationId"
              "timestamp", "tars:timestamp" ]

    type JsonLdWrapper<'T> =
        { [<JsonPropertyName("@context")>]
          Context: Map<string, string>
          [<JsonPropertyName("id")>]
          Id: Guid
          [<JsonPropertyName("correlationId")>]
          CorrelationId: CorrelationId
          [<JsonPropertyName("sender")>]
          Sender: MessageEndpoint
          [<JsonPropertyName("receiver")>]
          Receiver: MessageEndpoint option
          [<JsonPropertyName("performative")>]
          Performative: Performative
          [<JsonPropertyName("intent")>]
          Intent: AgentDomain option
          [<JsonPropertyName("constraints")>]
          Constraints: SemanticConstraints
          [<JsonPropertyName("ontology")>]
          Ontology: string option
          [<JsonPropertyName("language")>]
          Language: string
          [<JsonPropertyName("content")>]
          Content: 'T
          [<JsonPropertyName("timestamp")>]
          Timestamp: DateTime
          [<JsonPropertyName("metadata")>]
          Metadata: Map<string, string> }

    /// <summary>
    /// Serializes a SemanticMessage to a JSON-LD string.
    /// </summary>
    let toJsonLd (msg: SemanticMessage<'T>) : string =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        options.Converters.Add(JsonStringEnumConverter())

        let wrapper =
            { Context = jsonLdContext
              Id = msg.Id
              CorrelationId = msg.CorrelationId
              Sender = msg.Sender
              Receiver = msg.Receiver
              Performative = msg.Performative
              Intent = msg.Intent
              Constraints = msg.Constraints
              Ontology = msg.Ontology
              Language = msg.Language
              Content = msg.Content
              Timestamp = msg.Timestamp
              Metadata = msg.Metadata }

        JsonSerializer.Serialize(wrapper, options)

    /// <summary>
    /// Deserializes a JSON-LD string to a SemanticMessage.
    /// Note: This assumes the structure matches TARS SemanticMessage.
    /// Full JSON-LD parsing (expansion/compaction) is out of scope for now.
    /// </summary>
    let fromJsonLd<'T> (json: string) : SemanticMessage<'T> =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        options.Converters.Add(JsonStringEnumConverter())

        // We deserialize to a temporary structure to ignore @context
        // In a real JSON-LD processor, we would process the context.
        // However, since we want to return SemanticMessage<'T> which doesn't have Context,
        // we can just deserialize to SemanticMessage<'T> directly if we ignore extra fields?
        // No, JsonSerializer is strict by default or might ignore.
        // But SemanticMessage doesn't have @context field.
        // So deserializing to SemanticMessage directly will just ignore @context.

        JsonSerializer.Deserialize<SemanticMessage<'T>>(json, options)
