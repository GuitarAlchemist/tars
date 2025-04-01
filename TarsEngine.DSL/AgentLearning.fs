namespace TarsEngine.DSL

open System
open System.Collections.Generic
open Ast

/// Module containing the agent learning framework for the TARS DSL
module AgentLearning =
    /// Learning event type
    type LearningEventType =
        | TaskSuccess
        | TaskFailure
        | NewCapability
        | CapabilityImprovement
        | NewKnowledge
        | FeedbackReceived

    /// Learning event
    type LearningEvent = {
        Id: Guid
        AgentName: string
        EventType: LearningEventType
        Data: PropertyValue
        Timestamp: DateTime
    }

    /// Create a new learning event
    let createLearningEvent agentName eventType data =
        {
            Id = Guid.NewGuid()
            AgentName = agentName
            EventType = eventType
            Data = data
            Timestamp = DateTime.UtcNow
        }

    /// Learning model
    type LearningModel = {
        AgentName: string
        Capabilities: Map<string, float>
        Knowledge: Map<string, PropertyValue>
        SuccessRate: Map<string, float>
        LearningRate: float
        LastUpdated: DateTime
    }

    /// Create a new learning model
    let createLearningModel agentName =
        {
            AgentName = agentName
            Capabilities = Map.empty
            Knowledge = Map.empty
            SuccessRate = Map.empty
            LearningRate = 0.1
            LastUpdated = DateTime.UtcNow
        }

    /// Learning manager
    type LearningManager() =
        let models = Dictionary<string, LearningModel>()
        let events = List<LearningEvent>()
        let lockObj = Object()

        /// Get or create a learning model for an agent
        member this.GetModel(agentName: string) =
            lock lockObj (fun () ->
                if not (models.ContainsKey(agentName)) then
                    models.[agentName] <- createLearningModel agentName

                models.[agentName]
            )

        /// Record a learning event
        member this.RecordEvent(event: LearningEvent) =
            lock lockObj (fun () ->
                // Add the event to the list
                events.Add(event)

                // Update the learning model
                let model = this.GetModel(event.AgentName)
                let updatedModel = this.UpdateModel(model, event)
                models.[event.AgentName] <- updatedModel
            )

        /// Update a learning model based on an event
        member this.UpdateModel(model: LearningModel, event: LearningEvent) =
            match event.EventType with
            | TaskSuccess ->
                // Extract task name from event data
                let taskName =
                    match event.Data with
                    | ObjectValue props ->
                        match props.TryFind("taskName") with
                        | Some(StringValue name) -> name
                        | _ -> "unknown"
                    | _ -> "unknown"

                // Update success rate for the task
                let currentRate =
                    match model.SuccessRate.TryFind(taskName) with
                    | Some rate -> rate
                    | None -> 0.0

                let newRate = currentRate + model.LearningRate * (1.0 - currentRate)

                { model with
                    SuccessRate = model.SuccessRate.Add(taskName, newRate)
                    LastUpdated = DateTime.UtcNow }

            | TaskFailure ->
                // Extract task name from event data
                let taskName =
                    match event.Data with
                    | ObjectValue props ->
                        match props.TryFind("taskName") with
                        | Some(StringValue name) -> name
                        | _ -> "unknown"
                    | _ -> "unknown"

                // Update success rate for the task
                let currentRate =
                    match model.SuccessRate.TryFind(taskName) with
                    | Some rate -> rate
                    | None -> 0.0

                let newRate = currentRate - model.LearningRate * currentRate

                { model with
                    SuccessRate = model.SuccessRate.Add(taskName, newRate)
                    LastUpdated = DateTime.UtcNow }

            | NewCapability ->
                // Extract capability name and level from event data
                let (capabilityName, capabilityLevel) =
                    match event.Data with
                    | ObjectValue props ->
                        let name =
                            match props.TryFind("capabilityName") with
                            | Some(StringValue name) -> name
                            | _ -> "unknown"

                        let level =
                            match props.TryFind("capabilityLevel") with
                            | Some(NumberValue level) -> level
                            | _ -> 0.1

                        (name, level)
                    | _ -> ("unknown", 0.1)

                // Add the new capability
                { model with
                    Capabilities = model.Capabilities.Add(capabilityName, capabilityLevel)
                    LastUpdated = DateTime.UtcNow }

            | CapabilityImprovement ->
                // Extract capability name and improvement from event data
                let (capabilityName, improvement) =
                    match event.Data with
                    | ObjectValue props ->
                        let name =
                            match props.TryFind("capabilityName") with
                            | Some(StringValue name) -> name
                            | _ -> "unknown"

                        let improvement =
                            match props.TryFind("improvement") with
                            | Some(NumberValue improvement) -> improvement
                            | _ -> 0.1

                        (name, improvement)
                    | _ -> ("unknown", 0.1)

                // Update the capability level
                let currentLevel =
                    match model.Capabilities.TryFind(capabilityName) with
                    | Some level -> level
                    | None -> 0.0

                let newLevel = currentLevel + improvement

                { model with
                    Capabilities = model.Capabilities.Add(capabilityName, newLevel)
                    LastUpdated = DateTime.UtcNow }

            | NewKnowledge ->
                // Extract knowledge key and value from event data
                let (key, value) =
                    match event.Data with
                    | ObjectValue props ->
                        let key =
                            match props.TryFind("key") with
                            | Some(StringValue key) -> key
                            | _ -> "unknown"

                        let value =
                            match props.TryFind("value") with
                            | Some value -> value
                            | None -> StringValue("unknown")

                        (key, value)
                    | _ -> ("unknown", StringValue("unknown"))

                // Add the new knowledge
                { model with
                    Knowledge = model.Knowledge.Add(key, value)
                    LastUpdated = DateTime.UtcNow }

            | FeedbackReceived ->
                // Extract feedback data from event data
                match event.Data with
                | ObjectValue props ->
                    // Update learning rate based on feedback
                    let learningRate =
                        match props.TryFind("adjustLearningRate") with
                        | Some(NumberValue adjustment) ->
                            max 0.01 (min 1.0 (model.LearningRate + adjustment))
                        | _ -> model.LearningRate

                    // Update capabilities based on feedback
                    let capabilities =
                        match props.TryFind("capabilityAdjustments") with
                        | Some(ObjectValue adjustments) ->
                            adjustments |> Map.fold (fun (caps: Map<string, float>) (name: string) (value: PropertyValue) ->
                                match value with
                                | NumberValue adjustment ->
                                    let currentLevel =
                                        match caps.TryFind(name) with
                                        | Some level -> level
                                        | None -> 0.0

                                    let newLevel = max 0.0 (min 1.0 (currentLevel + adjustment))
                                    caps.Add(name, newLevel)
                                | _ -> caps
                            ) model.Capabilities
                        | _ -> model.Capabilities

                    { model with
                        LearningRate = learningRate
                        Capabilities = capabilities
                        LastUpdated = DateTime.UtcNow }
                | _ -> model

        /// Get the capability level for an agent
        member this.GetCapabilityLevel(agentName: string, capabilityName: string) =
            let model = this.GetModel(agentName)

            match model.Capabilities.TryFind(capabilityName) with
            | Some level -> level
            | None -> 0.0

        /// Get the success rate for a task
        member this.GetTaskSuccessRate(agentName: string, taskName: string) =
            let model = this.GetModel(agentName)

            match model.SuccessRate.TryFind(taskName) with
            | Some rate -> rate
            | None -> 0.0

        /// Get knowledge for an agent
        member this.GetKnowledge(agentName: string, key: string) =
            let model = this.GetModel(agentName)

            match model.Knowledge.TryFind(key) with
            | Some value -> Some value
            | None -> None

        /// Get all learning events for an agent
        member this.GetEvents(agentName: string) =
            events |> Seq.filter (fun e -> e.AgentName = agentName) |> Seq.toList

        /// Get learning statistics for an agent
        member this.GetStatistics(agentName: string) =
            let model = this.GetModel(agentName)
            let agentEvents = this.GetEvents(agentName)

            let eventCounts =
                agentEvents
                |> List.groupBy (fun e -> e.EventType)
                |> List.map (fun (eventType, events) -> (eventType, events.Length))
                |> Map.ofList

            let avgCapabilityLevel =
                if model.Capabilities.Count = 0 then 0.0
                else
                    model.Capabilities |> Map.toSeq |> Seq.averageBy snd

            let avgSuccessRate =
                if model.SuccessRate.Count = 0 then 0.0
                else
                    model.SuccessRate |> Map.toSeq |> Seq.averageBy snd

            ObjectValue(Map.empty
                .Add("agentName", StringValue(agentName))
                .Add("capabilityCount", NumberValue(float model.Capabilities.Count))
                .Add("knowledgeCount", NumberValue(float model.Knowledge.Count))
                .Add("avgCapabilityLevel", NumberValue(avgCapabilityLevel))
                .Add("avgSuccessRate", NumberValue(avgSuccessRate))
                .Add("learningRate", NumberValue(model.LearningRate))
                .Add("eventCounts", ObjectValue(eventCounts |> Map.toList |> List.map (fun (k, v) -> (k.ToString(), NumberValue(float v))) |> Map.ofList))
                .Add("lastUpdated", StringValue(model.LastUpdated.ToString("o"))))

    /// Global learning manager
    let learningManager = LearningManager()

    /// Record a task success
    let recordTaskSuccess agentName taskName =
        let data = ObjectValue(Map.empty.Add("taskName", StringValue(taskName)))
        let event = createLearningEvent agentName LearningEventType.TaskSuccess data
        learningManager.RecordEvent(event)

    /// Record a task failure
    let recordTaskFailure agentName taskName errorMessage =
        let data = ObjectValue(Map.empty
            .Add("taskName", StringValue(taskName))
            .Add("errorMessage", StringValue(errorMessage)))
        let event = createLearningEvent agentName LearningEventType.TaskFailure data
        learningManager.RecordEvent(event)

    /// Add a new capability
    let addCapability agentName capabilityName capabilityLevel =
        let data = ObjectValue(Map.empty
            .Add("capabilityName", StringValue(capabilityName))
            .Add("capabilityLevel", NumberValue(capabilityLevel)))
        let event = createLearningEvent agentName LearningEventType.NewCapability data
        learningManager.RecordEvent(event)

    /// Improve a capability
    let improveCapability agentName capabilityName improvement =
        let data = ObjectValue(Map.empty
            .Add("capabilityName", StringValue(capabilityName))
            .Add("improvement", NumberValue(improvement)))
        let event = createLearningEvent agentName LearningEventType.CapabilityImprovement data
        learningManager.RecordEvent(event)

    /// Add new knowledge
    let addKnowledge agentName key value =
        let data = ObjectValue(Map.empty
            .Add("key", StringValue(key))
            .Add("value", value))
        let event = createLearningEvent agentName LearningEventType.NewKnowledge data
        learningManager.RecordEvent(event)

    /// Provide feedback
    let provideFeedback agentName learningRateAdjustment capabilityAdjustments =
        let data = ObjectValue(Map.empty
            .Add("adjustLearningRate", NumberValue(learningRateAdjustment))
            .Add("capabilityAdjustments", ObjectValue(capabilityAdjustments)))
        let event = createLearningEvent agentName LearningEventType.FeedbackReceived data
        learningManager.RecordEvent(event)

    /// Get agent statistics
    let getAgentStatistics agentName =
        learningManager.GetStatistics(agentName)
