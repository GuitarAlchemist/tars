namespace Tars.Tests

open System
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Kernel

type KernelTests(output: ITestOutputHelper) =

    [<Fact>]
    member _.``Can create and register agent``() =
        output.WriteLine("Starting test: Can create and register agent")
        // Arrange
        let agentId = Guid.NewGuid()
        let name = "TestAgent"
        let model = "gpt-4-test"
        let prompt = "You are a test agent"
        let tools = []
        output.WriteLine($"Creating agent '{name}' with ID {agentId}")

        // Act
        let agent = AgentFactory.create agentId name "0.1.0" model prompt tools []
        let registry = AgentRegistry()
        output.WriteLine("Registering agent...")
        registry.Register(agent)

        // Assert
        let retrieved =
            (registry :> IAgentRegistry).GetAgent(AgentId agentId) |> Async.RunSynchronously

        Assert.True(retrieved.IsSome)
        Assert.Equal(AgentId agentId, retrieved.Value.Id)
        Assert.Equal(name, retrieved.Value.Name)
        output.WriteLine("Agent registered successfully.")

    [<Fact>]
    member _.``Can update agent state``() =
        output.WriteLine("Starting test: Can update agent state")
        // Arrange
        let agentId = Guid.NewGuid()
        let agent = AgentFactory.create agentId "Updater" "0.1.0" "model" "prompt" [] []
        let registry = AgentRegistry()
        registry.Register(agent)

        let newState = Error "Something happened"
        let updatedAgent = { agent with State = newState }
        output.WriteLine($"Updating agent {agentId} state to: {newState}")

        // Act
        registry.Register(updatedAgent) // Register acts as AddOrUpdate

        // Assert
        let storedAgent =
            (registry :> IAgentRegistry).GetAgent(AgentId agentId)
            |> Async.RunSynchronously
            |> Option.get

        Assert.Equal(newState, storedAgent.State)
        output.WriteLine("Agent state updated verified.")

    [<Fact>]
    member _.``Receive message adds to memory``() =
        output.WriteLine("Starting test: Receive message adds to memory")
        // Arrange
        let agent =
            AgentFactory.create (Guid.NewGuid()) "Receiver" "0.1.0" "model" "prompt" [] []

        let msg =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.System
              Receiver = Some(MessageEndpoint.Agent agent.Id)
              Performative = Performative.Inform
              Intent = None
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = "Hello"
              Timestamp = DateTime.UtcNow
              Metadata = Map.empty }

        output.WriteLine($"Sending message {msg.Id} to agent {agent.Id}")

        // Act
        let updatedAgent = agent.ReceiveMessage(msg)

        // Assert
        Assert.Single(updatedAgent.Memory) |> ignore
        Assert.Equal(msg, updatedAgent.Memory.Head)
        output.WriteLine("Message received and stored in memory.")
