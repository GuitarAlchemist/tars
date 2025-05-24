# DecisionService.fs Implementation Steps

## File Setup
1. Create file `TarsEngine.FSharp.Core/Consciousness/Decision/Services/DecisionService.fs`
2. Add namespace declaration:
   ```fsharp
   namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services
   ```
3. Add imports:
   ```fsharp
   open System
   open System.Collections.Generic
   open System.Threading.Tasks
   open Microsoft.Extensions.Logging
   open TarsEngine.FSharp.Core.Consciousness.Core
   open TarsEngine.FSharp.Core.Consciousness.Decision
   ```

## Class Definition
1. Define the DecisionService class with logger parameter:
   ```fsharp
   type DecisionService(logger: ILogger<DecisionService>) =
   ```
2. Add in-memory storage for decisions:
   ```fsharp
   // In-memory storage for decisions
   let decisions = Dictionary<Guid, Decision>()
   ```

## CreateDecision Method
1. Define method signature:
   ```fsharp
   member _.CreateDecision(name: string, description: string, type': DecisionType, ?priority: DecisionPriority, ?options: DecisionOption list, ?criteria: DecisionCriterion list, ?constraints: DecisionConstraint list, ?deadline: DateTime, ?context: string) =
   ```
2. Create task block:
   ```fsharp
   task {
       // Implementation here
   }
   ```
3. Generate new ID and get current time:
   ```fsharp
   let id = Guid.NewGuid()
   let now = DateTime.Now
   ```
4. Set default values for optional parameters:
   ```fsharp
   let priorityValue = defaultArg priority DecisionPriority.Medium
   let optionsValue = defaultArg options []
   let criteriaValue = defaultArg criteria []
   let constraintsValue = defaultArg constraints []
   ```
5. Create decision record:
   ```fsharp
   let decision = {
       Id = id
       Name = name
       Description = description
       Type = type'
       Status = DecisionStatus.Pending
       Priority = priorityValue
       Options = optionsValue
       Criteria = criteriaValue
       Constraints = constraintsValue
       SelectedOption = None
       CreationTime = now
       Deadline = deadline
       CompletionTime = None
       AssociatedEmotions = []
       Context = context
       Justification = None
       Metadata = Map.empty
   }
   ```
6. Add to storage and log:
   ```fsharp
   decisions.Add(id, decision)
   logger.LogInformation("Created decision {DecisionId} with name {DecisionName}", id, name)
   ```
7. Return the decision:
   ```fsharp
   return decision
   ```

## GetDecision Method
1. Define method signature:
   ```fsharp
   member _.GetDecision(id: Guid) =
   ```
2. Create task block:
   ```fsharp
   task {
       // Implementation here
   }
   ```
3. Check if decision exists and return:
   ```fsharp
   if decisions.ContainsKey(id) then
       return Some decisions.[id]
   else
       logger.LogWarning("Decision {DecisionId} not found", id)
       return None
   ```

## GetAllDecisions Method
1. Define method signature:
   ```fsharp
   member _.GetAllDecisions() =
   ```
2. Create task block:
   ```fsharp
   task {
       // Implementation here
   }
   ```
3. Return all decisions:
   ```fsharp
   return decisions.Values |> Seq.toList
   ```

## UpdateDecision Method
1. Define method signature:
   ```fsharp
   member _.UpdateDecision(id: Guid, ?description: string, ?status: DecisionStatus, ?priority: DecisionPriority, ?selectedOption: Guid, ?deadline: DateTime, ?context: string, ?justification: string) =
   ```
2. Create task block:
   ```fsharp
   task {
       // Implementation here
   }
   ```
3. Check if decision exists:
   ```fsharp
   if decisions.ContainsKey(id) then
       let decision = decisions.[id]
       
       let updatedDecision = {
           decision with
               Description = defaultArg description decision.Description
               Status = defaultArg status decision.Status
               Priority = defaultArg priority decision.Priority
               SelectedOption = 
                   match selectedOption with
                   | Some option -> Some option
                   | None -> decision.SelectedOption
               Deadline = 
                   match deadline with
                   | Some date -> Some date
                   | None -> decision.Deadline
               Context = 
                   match context with
                   | Some ctx -> Some ctx
                   | None -> decision.Context
               Justification = 
                   match justification with
                   | Some just -> Some just
                   | None -> decision.Justification
       }
       
       decisions.[id] <- updatedDecision
       logger.LogInformation("Updated decision {DecisionId}", id)
       
       return updatedDecision
   else
       logger.LogWarning("Decision {DecisionId} not found for update", id)
       return failwith $"Decision {id} not found"
   ```

## DeleteDecision Method
1. Define method signature:
   ```fsharp
   member _.DeleteDecision(id: Guid) =
   ```
2. Create task block:
   ```fsharp
   task {
       // Implementation here
   }
   ```
3. Check if decision exists and delete:
   ```fsharp
   if decisions.ContainsKey(id) then
       decisions.Remove(id) |> ignore
       logger.LogInformation("Deleted decision {DecisionId}", id)
       return true
   else
       logger.LogWarning("Decision {DecisionId} not found for deletion", id)
       return false
   ```

## AddOption Method
1. Define method signature:
   ```fsharp
   member _.AddOption(decisionId: Guid, name: string, description: string, ?pros: string list, ?cons: string list) =
   ```
2. Create task block:
   ```fsharp
   task {
       // Implementation here
   }
   ```
3. Check if decision exists:
   ```fsharp
   if decisions.ContainsKey(decisionId) then
       let decision = decisions.[decisionId]
       
       let optionId = Guid.NewGuid()
       let option = {
           Id = optionId
           Name = name
           Description = description
           Pros = defaultArg pros []
           Cons = defaultArg cons []
           Score = None
           Rank = None
           Metadata = Map.empty
       }
       
       let updatedDecision = {
           decision with
               Options = option :: decision.Options
       }
       
       decisions.[decisionId] <- updatedDecision
       logger.LogInformation("Added option {OptionId} to decision {DecisionId}", optionId, decisionId)
       
       return updatedDecision
   else
       logger.LogWarning("Decision {DecisionId} not found for adding option", decisionId)
       return failwith $"Decision {decisionId} not found"
   ```

## Interface Implementation
1. Add interface implementation:
   ```fsharp
   interface IDecisionService with
       member this.CreateDecision(name, description, type', ?priority, ?options, ?criteria, ?constraints, ?deadline, ?context) = 
           this.CreateDecision(name, description, type', ?priority = priority, ?options = options, ?criteria = criteria, ?constraints = constraints, ?deadline = deadline, ?context = context)
       
       member this.GetDecision(id) = this.GetDecision(id)
       
       member this.GetAllDecisions() = this.GetAllDecisions()
       
       member this.UpdateDecision(id, ?description, ?status, ?priority, ?selectedOption, ?deadline, ?context, ?justification) = 
           this.UpdateDecision(id, ?description = description, ?status = status, ?priority = priority, ?selectedOption = selectedOption, ?deadline = deadline, ?context = context, ?justification = justification)
       
       member this.DeleteDecision(id) = this.DeleteDecision(id)
       
       member this.AddOption(decisionId, name, description, ?pros, ?cons) = 
           this.AddOption(decisionId, name, description, ?pros = pros, ?cons = cons)
       
       // Add remaining interface implementations
   ```
