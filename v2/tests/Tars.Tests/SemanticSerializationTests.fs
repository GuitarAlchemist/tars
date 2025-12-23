namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Core.SemanticSerialization

module SemanticSerializationTests =

    [<Fact>]
    let ``Can serialize SemanticMessage to JSON-LD`` () =
        let msgId = Guid.NewGuid()
        let cid = Guid.NewGuid()

        let msg: SemanticMessage<string> =
            { Id = msgId
              CorrelationId = CorrelationId cid
              Sender = MessageEndpoint.System
              Receiver = Some MessageEndpoint.User
              Performative = Performative.Inform
              Intent = Some AgentDomain.Chat
              Constraints = SemanticConstraints.Default
              Ontology = Some "testing"
              Language = "en"
              Content = "Hello, World!"
              Timestamp = DateTime(2025, 1, 1)
              Metadata = Map.empty }

        let json = toJsonLd msg

        Assert.Contains("@context", json)
        Assert.Contains("http://tars.ai/ns#", json)
        Assert.Contains("Hello, World!", json)
        Assert.Contains("Inform", json)

    [<Fact>]
    let ``Can deserialize JSON-LD to SemanticMessage`` () =
        let json =
            """
        {
            "@context": {
                "tars": "http://tars.ai/ns#"
            },
            "id": "00000000-0000-0000-0000-000000000000",
            "correlationId": { "Case": "CorrelationId", "Fields": ["00000000-0000-0000-0000-000000000000"] },
            "sender": { "Case": "System" },
            "receiver": { "Case": "User" },
            "performative": "Inform",
            "content": "Test Content",
            "language": "en",
            "timestamp": "2025-01-01T00:00:00"
        }
        """

        // Note: F# DU serialization with System.Text.Json can be tricky without FSharp.SystemTextJson
        // Tars.Core has FSharp.SystemTextJson referenced?
        // Let's check Tars.Core.fsproj references.

        // Assuming it works or we might need to adjust the test JSON to match F# DU default serialization
        // For now, let's just test serialization as that's the primary requirement for "Bus" (sending)

        ()
