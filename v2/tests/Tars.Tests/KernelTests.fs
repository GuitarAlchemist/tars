namespace Tars.Tests

open System
open Xunit
open Xunit.Abstractions
open Tars.Core

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
        let agent = Kernel.createAgent agentId name "0.1.0" model prompt tools
        let ctx = Kernel.init ()
        output.WriteLine("Registering agent...")
        let updatedCtx = Kernel.registerAgent agent ctx

        // Assert
        Assert.Equal(AgentId agentId, agent.Id)
        Assert.Equal(name, agent.Name)
        Assert.True(updatedCtx.Agents.ContainsKey(AgentId agentId))
        Assert.Equal(agent, updatedCtx.Agents[AgentId agentId])
        output.WriteLine("Agent registered successfully.")

    [<Fact>]
    member _.``Can update agent state``() =
        output.WriteLine("Starting test: Can update agent state")
        // Arrange
        let agentId = Guid.NewGuid()
        let agent = Kernel.createAgent agentId "Updater" "0.1.0" "model" "prompt" []
        let ctx = Kernel.init () |> Kernel.registerAgent agent

        let newState = Error "Something happened"
        let updatedAgent = { agent with State = newState }
        output.WriteLine($"Updating agent {agentId} state to: {newState}")

        // Act
        let updatedCtx = Kernel.updateAgent updatedAgent ctx

        // Assert
        let storedAgent = updatedCtx.Agents[AgentId agentId]
        Assert.Equal(newState, storedAgent.State)
        output.WriteLine("Agent state updated verified.")

    [<Fact>]
    member _.``Receive message adds to memory``() =
        output.WriteLine("Starting test: Receive message adds to memory")
        // Arrange
        let agent =
            Kernel.createAgent (Guid.NewGuid()) "Receiver" "0.1.0" "model" "prompt" []

        let msg =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.System
              Receiver = Some(MessageEndpoint.Agent agent.Id)
              Performative = Performative.Inform
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = "Hello"
              Timestamp = DateTime.UtcNow
              Metadata = Map.empty }

        output.WriteLine($"Sending message {msg.Id} to agent {agent.Id}")

        // Act
        let updatedAgent = Kernel.receiveMessage msg agent

        // Assert
        Assert.Single(updatedAgent.Memory) |> ignore
        Assert.Equal(msg, updatedAgent.Memory.Head)
        output.WriteLine("Message received and stored in memory.")
