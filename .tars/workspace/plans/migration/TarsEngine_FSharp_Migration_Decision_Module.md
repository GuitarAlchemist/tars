# TARS Engine Migration to F# - Decision Module Detailed Tasks

## Decision Module Implementation

### Types (Completed) ✅
- [x] Create DecisionType discriminated union
- [x] Create DecisionStatus discriminated union
- [x] Create DecisionPriority discriminated union
- [x] Create DecisionOption record type
- [x] Create DecisionCriterion record type
- [x] Create DecisionConstraint record type
- [x] Create Decision record type
- [x] Create DecisionEvaluation record type
- [x] Create DecisionQuery record type
- [x] Create DecisionQueryResult record type

### IDecisionService Interface (Completed) ✅
- [x] Define CreateDecision method signature
- [x] Define GetDecision method signature
- [x] Define GetAllDecisions method signature
- [x] Define UpdateDecision method signature
- [x] Define DeleteDecision method signature
- [x] Define AddOption method signature
- [x] Define UpdateOption method signature
- [x] Define RemoveOption method signature
- [x] Define AddCriterion method signature
- [x] Define UpdateCriterion method signature
- [x] Define RemoveCriterion method signature
- [x] Define ScoreOption method signature
- [x] Define AddConstraint method signature
- [x] Define UpdateConstraint method signature
- [x] Define RemoveConstraint method signature
- [x] Define AddEmotionToDecision method signature
- [x] Define EvaluateDecision method signature
- [x] Define MakeDecision method signature
- [x] Define FindDecisions method signature

### DecisionService Implementation 🔄

#### Setup
- [ ] Create DecisionService.fs file
- [ ] Define namespace and imports
- [ ] Create in-memory storage for decisions (Dictionary)
- [ ] Define DecisionService class with logger parameter

#### CreateDecision Method
- [ ] Define CreateDecision method signature
- [ ] Generate new GUID for decision ID
- [ ] Get current timestamp for creation time
- [ ] Create empty lists for options, criteria, constraints, emotions
- [ ] Create decision record with provided and default values
- [ ] Add decision to in-memory storage
- [ ] Log decision creation with ID and name
- [ ] Return created decision as Task

#### GetDecision Method
- [ ] Define GetDecision method signature
- [ ] Check if decision exists in storage using ContainsKey
- [ ] If exists, return Some decision wrapped in Task
- [ ] If not exists, log warning and return None wrapped in Task

#### GetAllDecisions Method
- [ ] Define GetAllDecisions method signature
- [ ] Convert Dictionary values to list
- [ ] Return list wrapped in Task

#### UpdateDecision Method
- [ ] Define UpdateDecision method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Create updated decision with new values (using defaultArg for optional parameters)
- [ ] Update decision in storage
- [ ] Log update with decision ID
- [ ] Return updated decision wrapped in Task

#### DeleteDecision Method
- [ ] Define DeleteDecision method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return false
- [ ] Remove decision from storage
- [ ] Log deletion with decision ID
- [ ] Return true wrapped in Task

#### AddOption Method
- [ ] Define AddOption method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Generate new GUID for option ID
- [ ] Create new option record with provided values
- [ ] Get existing decision from storage
- [ ] Create updated decision with new option added to options list
- [ ] Update decision in storage
- [ ] Log option addition with decision ID and option name
- [ ] Return updated decision wrapped in Task

#### UpdateOption Method
- [ ] Define UpdateOption method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Find option in decision's options list
- [ ] If option not found, log warning and return error
- [ ] Create updated option with new values
- [ ] Create updated options list with updated option
- [ ] Create updated decision with new options list
- [ ] Update decision in storage
- [ ] Log option update with decision ID and option ID
- [ ] Return updated decision wrapped in Task

#### RemoveOption Method
- [ ] Define RemoveOption method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Create updated options list without specified option
- [ ] Create updated decision with new options list
- [ ] Update decision in storage
- [ ] Log option removal with decision ID and option ID
- [ ] Return updated decision wrapped in Task

#### AddCriterion Method
- [ ] Define AddCriterion method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Generate new GUID for criterion ID
- [ ] Create new criterion record with provided values and empty scores
- [ ] Get existing decision from storage
- [ ] Create updated decision with new criterion added to criteria list
- [ ] Update decision in storage
- [ ] Log criterion addition with decision ID and criterion name
- [ ] Return updated decision wrapped in Task

#### UpdateCriterion Method
- [ ] Define UpdateCriterion method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Find criterion in decision's criteria list
- [ ] If criterion not found, log warning and return error
- [ ] Create updated criterion with new values
- [ ] Create updated criteria list with updated criterion
- [ ] Create updated decision with new criteria list
- [ ] Update decision in storage
- [ ] Log criterion update with decision ID and criterion ID
- [ ] Return updated decision wrapped in Task

#### RemoveCriterion Method
- [ ] Define RemoveCriterion method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Create updated criteria list without specified criterion
- [ ] Create updated decision with new criteria list
- [ ] Update decision in storage
- [ ] Log criterion removal with decision ID and criterion ID
- [ ] Return updated decision wrapped in Task

#### ScoreOption Method
- [ ] Define ScoreOption method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Find criterion in decision's criteria list
- [ ] If criterion not found, log warning and return error
- [ ] Find option in decision's options list
- [ ] If option not found, log warning and return error
- [ ] Create updated criterion with new score for option
- [ ] Create updated criteria list with updated criterion
- [ ] Create updated decision with new criteria list
- [ ] Update decision in storage
- [ ] Log option scoring with decision ID, criterion ID, and option ID
- [ ] Return updated decision wrapped in Task

#### AddConstraint Method
- [ ] Define AddConstraint method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Generate new GUID for constraint ID
- [ ] Create new constraint record with provided values
- [ ] Get existing decision from storage
- [ ] Create updated decision with new constraint added to constraints list
- [ ] Update decision in storage
- [ ] Log constraint addition with decision ID and constraint name
- [ ] Return updated decision wrapped in Task

#### UpdateConstraint Method
- [ ] Define UpdateConstraint method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Find constraint in decision's constraints list
- [ ] If constraint not found, log warning and return error
- [ ] Create updated constraint with new values
- [ ] Create updated constraints list with updated constraint
- [ ] Create updated decision with new constraints list
- [ ] Update decision in storage
- [ ] Log constraint update with decision ID and constraint ID
- [ ] Return updated decision wrapped in Task

#### RemoveConstraint Method
- [ ] Define RemoveConstraint method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Create updated constraints list without specified constraint
- [ ] Create updated decision with new constraints list
- [ ] Update decision in storage
- [ ] Log constraint removal with decision ID and constraint ID
- [ ] Return updated decision wrapped in Task

#### AddEmotionToDecision Method
- [ ] Define AddEmotionToDecision method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Create updated decision with new emotion added to emotions list
- [ ] Update decision in storage
- [ ] Log emotion addition with decision ID and emotion category
- [ ] Return updated decision wrapped in Task

#### EvaluateDecision Method
- [ ] Define EvaluateDecision method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Calculate weighted scores for each option based on criteria
- [ ] Identify strengths (high scores) for each option
- [ ] Identify weaknesses (low scores) for each option
- [ ] Identify opportunities (potential improvements)
- [ ] Identify threats (potential risks)
- [ ] Create evaluation record with calculated data
- [ ] Log evaluation with decision ID
- [ ] Return evaluation wrapped in Task

#### MakeDecision Method
- [ ] Define MakeDecision method signature
- [ ] Check if decision exists in storage
- [ ] If not exists, log warning and return error
- [ ] Get existing decision from storage
- [ ] Calculate weighted scores for each option based on criteria
- [ ] Find option with highest score
- [ ] Create updated decision with:
  - [ ] Status set to Completed
  - [ ] SelectedOption set to best option ID
  - [ ] CompletionTime set to current time
  - [ ] Justification explaining the selection
- [ ] Update decision in storage
- [ ] Log decision making with decision ID and selected option
- [ ] Return updated decision wrapped in Task

#### FindDecisions Method
- [ ] Define FindDecisions method signature
- [ ] Get start time for performance tracking
- [ ] Get all decisions from storage
- [ ] Filter by name pattern if specified
- [ ] Filter by types if specified
- [ ] Filter by statuses if specified
- [ ] Filter by priorities if specified
- [ ] Filter by minimum creation time if specified
- [ ] Filter by maximum creation time if specified
- [ ] Limit results if specified
- [ ] Get end time for performance tracking
- [ ] Calculate execution time
- [ ] Create query result with filtered decisions and execution time
- [ ] Log search with query parameters and result count
- [ ] Return query result wrapped in Task

#### Interface Implementation
- [ ] Implement IDecisionService interface
- [ ] Map all interface methods to class methods

### DecisionService Unit Tests

#### Test Setup
- [ ] Create DecisionServiceTests.fs file
- [ ] Define namespace and imports
- [ ] Create MockLogger for testing
- [ ] Define helper functions for creating test decisions

#### CreateDecision Tests
- [ ] Test creating decision with minimal parameters
- [ ] Test creating decision with all parameters
- [ ] Test creating decision with options
- [ ] Test creating decision with criteria
- [ ] Test creating decision with constraints

#### GetDecision Tests
- [ ] Test getting existing decision
- [ ] Test getting non-existent decision

#### GetAllDecisions Tests
- [ ] Test getting all decisions when empty
- [ ] Test getting all decisions with multiple decisions

#### UpdateDecision Tests
- [ ] Test updating decision description
- [ ] Test updating decision status
- [ ] Test updating decision priority
- [ ] Test updating decision selected option
- [ ] Test updating decision deadline
- [ ] Test updating decision context
- [ ] Test updating decision justification
- [ ] Test updating non-existent decision

#### DeleteDecision Tests
- [ ] Test deleting existing decision
- [ ] Test deleting non-existent decision

#### AddOption Tests
- [ ] Test adding option to decision
- [ ] Test adding option to non-existent decision

#### UpdateOption Tests
- [ ] Test updating existing option
- [ ] Test updating non-existent option
- [ ] Test updating option in non-existent decision

#### RemoveOption Tests
- [ ] Test removing existing option
- [ ] Test removing non-existent option
- [ ] Test removing option from non-existent decision

#### AddCriterion Tests
- [ ] Test adding criterion to decision
- [ ] Test adding criterion to non-existent decision

#### UpdateCriterion Tests
- [ ] Test updating existing criterion
- [ ] Test updating non-existent criterion
- [ ] Test updating criterion in non-existent decision

#### RemoveCriterion Tests
- [ ] Test removing existing criterion
- [ ] Test removing non-existent criterion
- [ ] Test removing criterion from non-existent decision

#### ScoreOption Tests
- [ ] Test scoring option for criterion
- [ ] Test scoring non-existent option
- [ ] Test scoring option for non-existent criterion
- [ ] Test scoring option in non-existent decision

#### AddConstraint Tests
- [ ] Test adding constraint to decision
- [ ] Test adding constraint to non-existent decision

#### UpdateConstraint Tests
- [ ] Test updating existing constraint
- [ ] Test updating non-existent constraint
- [ ] Test updating constraint in non-existent decision

#### RemoveConstraint Tests
- [ ] Test removing existing constraint
- [ ] Test removing non-existent constraint
- [ ] Test removing constraint from non-existent decision

#### AddEmotionToDecision Tests
- [ ] Test adding emotion to decision
- [ ] Test adding emotion to non-existent decision

#### EvaluateDecision Tests
- [ ] Test evaluating decision with options and criteria
- [ ] Test evaluating decision without options
- [ ] Test evaluating decision without criteria
- [ ] Test evaluating non-existent decision

#### MakeDecision Tests
- [ ] Test making decision with options and criteria
- [ ] Test making decision without options
- [ ] Test making decision without criteria
- [ ] Test making non-existent decision

#### FindDecisions Tests
- [ ] Test finding decisions by name pattern
- [ ] Test finding decisions by type
- [ ] Test finding decisions by status
- [ ] Test finding decisions by priority
- [ ] Test finding decisions by creation time range
- [ ] Test finding decisions with limit
- [ ] Test finding decisions with no matches
