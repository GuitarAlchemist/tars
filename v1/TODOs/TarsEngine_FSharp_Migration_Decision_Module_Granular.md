# TARS Engine Migration to F# - Decision Module Ultra-Granular Tasks

## Decision Module Implementation

### Setup
- [ ] Create DecisionService.fs file
- [ ] Add namespace declaration for TarsEngine.FSharp.Core.Consciousness.Decision.Services
- [ ] Add System imports (System, System.Collections.Generic, System.Threading.Tasks)
- [ ] Add Microsoft.Extensions.Logging import
- [ ] Add TarsEngine.FSharp.Core.Consciousness.Core import
- [ ] Add TarsEngine.FSharp.Core.Consciousness.Decision import
- [ ] Define Dictionary<Guid, Decision> for in-memory storage
- [ ] Define DecisionService class with ILogger<DecisionService> parameter
- [ ] Initialize empty dictionary in constructor

### CreateDecision Method
- [ ] Define CreateDecision method signature with all parameters
- [ ] Generate new Guid using Guid.NewGuid()
- [ ] Get current timestamp using DateTime.Now
- [ ] Create empty list for options (defaultArg options [])
- [ ] Create empty list for criteria (defaultArg criteria [])
- [ ] Create empty list for constraints (defaultArg constraints [])
- [ ] Create empty list for emotions ([])
- [ ] Create empty map for metadata (Map.empty)
- [ ] Set initial status to Pending
- [ ] Create decision record with all fields
- [ ] Add decision to dictionary using Add method
- [ ] Log information using logger.LogInformation with decision ID and name
- [ ] Return decision wrapped in Task.FromResult

### GetDecision Method
- [ ] Define GetDecision method signature with modelId parameter
- [ ] Check if decision exists using ContainsKey method
- [ ] If exists, get decision from dictionary
- [ ] If exists, return Some decision wrapped in Task.FromResult
- [ ] If not exists, log warning using logger.LogWarning
- [ ] If not exists, return None wrapped in Task.FromResult

### GetAllDecisions Method
- [ ] Define GetAllDecisions method signature
- [ ] Get all values from dictionary using Values property
- [ ] Convert values to list using Seq.toList
- [ ] Return list wrapped in Task.FromResult

### UpdateDecision Method
- [ ] Define UpdateDecision method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Create updated description using defaultArg
- [ ] Create updated status using defaultArg
- [ ] Create updated priority using defaultArg
- [ ] Create updated selectedOption using defaultArg
- [ ] Create updated deadline using defaultArg
- [ ] Create updated context using defaultArg
- [ ] Create updated justification using defaultArg
- [ ] Create updated decision record with all fields
- [ ] Update decision in dictionary using indexer
- [ ] Log information about update
- [ ] Return updated decision wrapped in Task.FromResult

### DeleteDecision Method
- [ ] Define DeleteDecision method signature with id parameter
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return false wrapped in Task.FromResult
- [ ] Remove decision from dictionary using Remove method
- [ ] Log information about deletion
- [ ] Return true wrapped in Task.FromResult

### AddOption Method
- [ ] Define AddOption method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Generate new Guid for option using Guid.NewGuid()
- [ ] Create empty list for pros (defaultArg pros [])
- [ ] Create empty list for cons (defaultArg cons [])
- [ ] Create empty map for metadata (Map.empty)
- [ ] Create option record with all fields
- [ ] Get existing decision from dictionary
- [ ] Create updated options list by adding new option
- [ ] Create updated decision record with new options list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about option addition
- [ ] Return updated decision wrapped in Task.FromResult

### UpdateOption Method
- [ ] Define UpdateOption method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Find option in options list using List.tryFind
- [ ] If option not found, log warning and return error using failwith
- [ ] Create updated name using defaultArg
- [ ] Create updated description using defaultArg
- [ ] Create updated pros using defaultArg
- [ ] Create updated cons using defaultArg
- [ ] Create updated score using defaultArg
- [ ] Create updated rank using defaultArg
- [ ] Create updated option record with all fields
- [ ] Create updated options list by replacing old option with new option
- [ ] Create updated decision record with new options list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about option update
- [ ] Return updated decision wrapped in Task.FromResult

### RemoveOption Method
- [ ] Define RemoveOption method signature with decisionId and optionId parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Create updated options list by filtering out option with matching ID
- [ ] Create updated decision record with new options list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about option removal
- [ ] Return updated decision wrapped in Task.FromResult

### AddCriterion Method
- [ ] Define AddCriterion method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Generate new Guid for criterion using Guid.NewGuid()
- [ ] Create empty map for scores (Map.empty)
- [ ] Create empty map for metadata (Map.empty)
- [ ] Create criterion record with all fields
- [ ] Get existing decision from dictionary
- [ ] Create updated criteria list by adding new criterion
- [ ] Create updated decision record with new criteria list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about criterion addition
- [ ] Return updated decision wrapped in Task.FromResult

### UpdateCriterion Method
- [ ] Define UpdateCriterion method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Find criterion in criteria list using List.tryFind
- [ ] If criterion not found, log warning and return error using failwith
- [ ] Create updated name using defaultArg
- [ ] Create updated description using defaultArg
- [ ] Create updated weight using defaultArg
- [ ] Create updated criterion record with all fields
- [ ] Create updated criteria list by replacing old criterion with new criterion
- [ ] Create updated decision record with new criteria list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about criterion update
- [ ] Return updated decision wrapped in Task.FromResult

### RemoveCriterion Method
- [ ] Define RemoveCriterion method signature with decisionId and criterionId parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Create updated criteria list by filtering out criterion with matching ID
- [ ] Create updated decision record with new criteria list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about criterion removal
- [ ] Return updated decision wrapped in Task.FromResult

### ScoreOption Method
- [ ] Define ScoreOption method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Find criterion in criteria list using List.tryFind
- [ ] If criterion not found, log warning and return error using failwith
- [ ] Find option in options list using List.exists
- [ ] If option not found, log warning and return error using failwith
- [ ] Create updated scores map by adding or updating score for option
- [ ] Create updated criterion record with new scores map
- [ ] Create updated criteria list by replacing old criterion with new criterion
- [ ] Create updated decision record with new criteria list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about option scoring
- [ ] Return updated decision wrapped in Task.FromResult

### AddConstraint Method
- [ ] Define AddConstraint method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Generate new Guid for constraint using Guid.NewGuid()
- [ ] Create empty map for metadata (Map.empty)
- [ ] Create constraint record with all fields
- [ ] Get existing decision from dictionary
- [ ] Create updated constraints list by adding new constraint
- [ ] Create updated decision record with new constraints list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about constraint addition
- [ ] Return updated decision wrapped in Task.FromResult

### UpdateConstraint Method
- [ ] Define UpdateConstraint method signature with all parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Find constraint in constraints list using List.tryFind
- [ ] If constraint not found, log warning and return error using failwith
- [ ] Create updated name using defaultArg
- [ ] Create updated description using defaultArg
- [ ] Create updated type using defaultArg
- [ ] Create updated value using defaultArg
- [ ] Create updated isSatisfied using defaultArg
- [ ] Create updated constraint record with all fields
- [ ] Create updated constraints list by replacing old constraint with new constraint
- [ ] Create updated decision record with new constraints list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about constraint update
- [ ] Return updated decision wrapped in Task.FromResult

### RemoveConstraint Method
- [ ] Define RemoveConstraint method signature with decisionId and constraintId parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Create updated constraints list by filtering out constraint with matching ID
- [ ] Create updated decision record with new constraints list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about constraint removal
- [ ] Return updated decision wrapped in Task.FromResult

### AddEmotionToDecision Method
- [ ] Define AddEmotionToDecision method signature with decisionId and emotion parameters
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Create updated emotions list by adding new emotion
- [ ] Create updated decision record with new emotions list
- [ ] Update decision in dictionary using indexer
- [ ] Log information about emotion addition
- [ ] Return updated decision wrapped in Task.FromResult

### EvaluateDecision Method
- [ ] Define EvaluateDecision method signature with decisionId parameter
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Calculate total weight of all criteria
- [ ] Calculate weighted scores for each option based on criteria
- [ ] Identify strengths for each option (criteria with high scores)
- [ ] Identify weaknesses for each option (criteria with low scores)
- [ ] Identify opportunities (potential improvements)
- [ ] Identify threats (potential risks)
- [ ] Create evaluation record with calculated data
- [ ] Log information about evaluation
- [ ] Return evaluation wrapped in Task.FromResult

### MakeDecision Method
- [ ] Define MakeDecision method signature with decisionId parameter
- [ ] Check if decision exists using ContainsKey method
- [ ] If not exists, log warning and return error using failwith
- [ ] Get existing decision from dictionary
- [ ] Calculate total weight of all criteria
- [ ] Calculate weighted scores for each option based on criteria
- [ ] Find option with highest score
- [ ] Create justification string explaining the selection
- [ ] Create updated decision record with:
  - [ ] Status set to Completed
  - [ ] SelectedOption set to best option ID
  - [ ] CompletionTime set to current time
  - [ ] Justification set to explanation
- [ ] Update decision in dictionary using indexer
- [ ] Log information about decision making
- [ ] Return updated decision wrapped in Task.FromResult

### FindDecisions Method
- [ ] Define FindDecisions method signature with query parameter
- [ ] Get start time using DateTime.Now
- [ ] Get all decisions from dictionary using Values property
- [ ] Convert values to list using Seq.toList
- [ ] Filter by name pattern if specified using List.filter and String.Contains
- [ ] Filter by types if specified using List.filter and List.exists
- [ ] Filter by statuses if specified using List.filter and List.exists
- [ ] Filter by priorities if specified using List.filter and List.exists
- [ ] Filter by minimum creation time if specified using List.filter
- [ ] Filter by maximum creation time if specified using List.filter
- [ ] Limit results if specified using List.truncate
- [ ] Get end time using DateTime.Now
- [ ] Calculate execution time by subtracting start time from end time
- [ ] Create query result record with filtered decisions and execution time
- [ ] Log information about search
- [ ] Return query result wrapped in Task.FromResult

### Interface Implementation
- [ ] Add interface implementation section using interface IDecisionService with
- [ ] Implement member this.CreateDecision
- [ ] Implement member this.GetDecision
- [ ] Implement member this.GetAllDecisions
- [ ] Implement member this.UpdateDecision
- [ ] Implement member this.DeleteDecision
- [ ] Implement member this.AddOption
- [ ] Implement member this.UpdateOption
- [ ] Implement member this.RemoveOption
- [ ] Implement member this.AddCriterion
- [ ] Implement member this.UpdateCriterion
- [ ] Implement member this.RemoveCriterion
- [ ] Implement member this.ScoreOption
- [ ] Implement member this.AddConstraint
- [ ] Implement member this.UpdateConstraint
- [ ] Implement member this.RemoveConstraint
- [ ] Implement member this.AddEmotionToDecision
- [ ] Implement member this.EvaluateDecision
- [ ] Implement member this.MakeDecision
- [ ] Implement member this.FindDecisions
