module Tars.Tests.SpeechActsTests

open System
open Xunit
open Tars.Core

[<Fact>]
let ``determineProtocol maps Ask to RequestResponse`` () =
    let intent = AgentIntent.Ask "Question"
    let protocol = SpeechActs.determineProtocol intent
    Assert.Equal(InteractionProtocol.RequestResponse, protocol)

[<Fact>]
let ``determineProtocol maps Propose to ContractNet`` () =
    let intent = AgentIntent.Propose "Plan"
    let protocol = SpeechActs.determineProtocol intent
    Assert.Equal(InteractionProtocol.ContractNet, protocol)

[<Fact>]
let ``validateFlow allows Ask to Tell`` () =
    let correlationId = CorrelationId(Guid.NewGuid())
    let agent1 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))
    let agent2 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))

    let original =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent1
          To = Some agent2
          Intent = AgentIntent.Ask "Q"
          Domain = None
          Content = "Content"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let reply =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent2
          To = Some agent1
          Intent = AgentIntent.Tell "A"
          Domain = None
          Content = "Answer"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let result = SpeechActs.validateFlow original reply

    match result with
    | Result.Ok _ -> ()
    | Result.Error e -> Assert.Fail($"Expected Ok but got Error: {e}")

[<Fact>]
let ``validateFlow allows Propose to Accept with matching ID`` () =
    let correlationId = CorrelationId(Guid.NewGuid())
    let agent1 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))
    let agent2 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))

    let original =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent1
          To = Some agent2
          Intent = AgentIntent.Propose "Plan"
          Domain = None
          Content = "Content"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let acceptIntent = Accept(original.Id)

    let reply =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent2
          To = Some agent1
          Intent = acceptIntent
          Domain = None
          Content = "OK"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let result = SpeechActs.validateFlow original reply

    match result with
    | Result.Ok _ -> ()
    | Result.Error e -> Assert.Fail($"Expected Ok but got Error: {e}")

[<Fact>]
let ``validateFlow rejects Propose to Accept with wrong ID`` () =
    let correlationId = CorrelationId(Guid.NewGuid())
    let agent1 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))
    let agent2 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))

    let original =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent1
          To = Some agent2
          Intent = AgentIntent.Propose "Plan"
          Domain = None
          Content = "Content"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let reply =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent2
          To = Some agent1
          Intent = AgentIntent.Accept(Guid.NewGuid()) // Wrong ID
          Domain = None
          Content = "OK"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let result = SpeechActs.validateFlow original reply

    match result with
    | Result.Error _ -> ()
    | Result.Ok _ -> Assert.Fail("Expected Error but got Ok")

[<Fact>]
let ``validateFlow rejects Ask to Propose`` () =
    let correlationId = CorrelationId(Guid.NewGuid())
    let agent1 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))
    let agent2 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))

    let original =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent1
          To = Some agent2
          Intent = AgentIntent.Ask "Q"
          Domain = None
          Content = "Content"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let reply =
        { Id = Guid.NewGuid()
          CorrelationId = correlationId
          From = agent2
          To = Some agent1
          Intent = AgentIntent.Propose "P"
          Domain = None
          Content = "Content"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let result = SpeechActs.validateFlow original reply

    match result with
    | Result.Error _ -> ()
    | Result.Ok _ -> Assert.Fail("Expected Error but got Ok")

[<Fact>]
let ``validateFlow rejects CorrelationId mismatch`` () =
    let agent1 = MessageEndpoint.Agent(AgentId(Guid.NewGuid()))

    let original =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId(Guid.NewGuid())
          From = agent1
          To = None
          Intent = AgentIntent.Ask "Q"
          Domain = None
          Content = "C"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let reply =
        { Id = Guid.NewGuid()
          CorrelationId = CorrelationId(Guid.NewGuid()) // Different ID
          From = agent1
          To = None
          Intent = AgentIntent.Tell "A"
          Domain = None
          Content = "C"
          Timestamp = DateTime.UtcNow
          Metadata = Map.empty }

    let result = SpeechActs.validateFlow original reply

    match result with
    | Result.Error _ -> ()
    | Result.Ok _ -> Assert.Fail("Expected Error but got Ok")
