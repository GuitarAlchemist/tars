# TARS Engine F# Migration - Planning Module Granular Tasks

## Overview
This document provides a detailed breakdown of tasks for implementing the Planning module in F#. The Planning module is responsible for creating, executing, and monitoring execution plans for improvements.

## Planning Module Implementation

### Types.fs
- [x] Create ExecutionStepType discriminated union
  - [x] Define FileModification, FileCreation, FileDeletion, etc.
- [x] Create ExecutionStepStatus discriminated union
  - [x] Define Pending, InProgress, Completed, Failed, etc.
- [x] Create ExecutionMode discriminated union
  - [x] Define DryRun, Live
- [x] Create ExecutionEnvironment discriminated union
  - [x] Define Sandbox, Development, Testing, Production
- [x] Create ExecutionPlanStatus discriminated union
  - [x] Define Created, Scheduled, InProgress, etc.
- [x] Create LogLevel discriminated union
  - [x] Define Trace, Debug, Information, etc.
- [x] Create ValidationRule record type
  - [x] Define Name, Description, Expression, etc.
- [x] Create ExecutionStepResult record type
  - [x] Define ExecutionStepId, Status, IsSuccessful, etc.
- [x] Create ExecutionStep record type
  - [x] Define Id, Name, Description, Type, etc.
- [x] Create ExecutionLog record type
  - [x] Define Timestamp, Level, Message, etc.
- [x] Create ExecutionError record type
  - [x] Define Timestamp, Message, Source, etc.
- [x] Create ExecutionWarning record type
  - [x] Define Timestamp, Message, Source
- [x] Create ExecutionPermissions record type
  - [x] Define AllowFileSystem, AllowNetwork, etc.
- [x] Create ExecutionContext record type
  - [x] Define Id, ExecutionPlanId, ImprovementId, etc.
- [x] Create ExecutionPlanResult record type
  - [x] Define ExecutionPlanId, Status, IsSuccessful, etc.
- [x] Create ExecutionPlan record type
  - [x] Define Id, Name, Description, etc.
- [x] Create ExecutionStep module with helper functions
  - [x] Implement create function
  - [x] Implement isCompleted function
  - [x] Implement isSuccessful function
  - [x] Implement isReadyToExecute function
- [x] Create ExecutionContext module with helper functions
  - [x] Implement create function
  - [x] Implement addLog function
  - [x] Implement addError function
  - [x] Implement addWarning function
  - [x] Implement setVariable function
  - [x] Implement getVariable function
  - [x] Implement setState function
  - [x] Implement getState function
  - [x] Implement setMetric function
  - [x] Implement getMetric function
  - [x] Implement addAffectedFile function
  - [x] Implement addBackedUpFile function
  - [x] Implement addModifiedFile function
  - [x] Implement addCreatedFile function
  - [x] Implement addDeletedFile function
- [x] Create ExecutionPlan module with helper functions
  - [x] Implement create function
  - [x] Implement totalSteps function
  - [x] Implement completedSteps function
  - [x] Implement failedSteps function
  - [x] Implement progress function
  - [x] Implement isCompleted function
  - [x] Implement isSuccessful function
  - [x] Implement getNextStep function
  - [x] Implement getCurrentStep function
  - [x] Implement getDependencies function
  - [x] Implement getDependents function
  - [x] Implement validate function

### IPlanningService.fs
- [x] Create IPlanningService interface
  - [x] Define CreateExecutionPlan method
  - [x] Define GetExecutionPlan method
  - [x] Define GetAllExecutionPlans method
  - [x] Define UpdateExecutionPlan method
  - [x] Define DeleteExecutionPlan method
  - [x] Define ExecuteStep method
  - [x] Define ExecutePlan method
  - [x] Define ValidatePlan method
  - [x] Define MonitorPlan method
  - [x] Define AdaptPlan method
  - [x] Define RollbackStep method
  - [x] Define RollbackPlan method
  - [x] Define GetExecutionContext method
  - [x] Define UpdateExecutionContext method

### ExecutionPlanner.fs
- [x] Create ExecutionPlanner class
  - [x] Implement constructor with logger and dependencies
  - [x] Create in-memory storage for execution plans
  - [x] Create in-memory storage for execution contexts
  - [x] Implement CreateExecutionPlan method
    - [x] Generate new GUID for plan
    - [x] Create plan record with initial values
    - [x] Add plan to storage
    - [x] Log creation
    - [x] Return created plan
  - [x] Implement GetExecutionPlan method
    - [x] Check if plan exists in storage
    - [x] Return plan if found, None otherwise
    - [x] Log retrieval attempt
  - [x] Implement GetAllExecutionPlans method
    - [x] Return all plans from storage
  - [x] Implement UpdateExecutionPlan method
    - [x] Check if plan exists in storage
    - [x] Create updated plan with new values
    - [x] Update plan in storage
    - [x] Log update
    - [x] Return updated plan
  - [x] Implement DeleteExecutionPlan method
    - [x] Check if plan exists in storage
    - [x] Remove plan from storage
    - [x] Log deletion
    - [x] Return success/failure
  - [x] Implement ExecuteStep method
    - [x] Check if plan exists in storage
    - [x] Check if step exists in plan
    - [x] Check if step is ready to execute
    - [x] Update step status to InProgress
    - [x] Set step start time
    - [x] Execute step action
    - [x] Handle step result
    - [x] Update step status to Completed/Failed
    - [x] Set step completion time
    - [x] Calculate step duration
    - [x] Update plan in storage
    - [x] Log execution
    - [x] Return step result
  - [x] Implement ExecutePlan method
    - [x] Check if plan exists in storage
    - [x] Create execution context
    - [x] Update plan status to InProgress
    - [x] Set plan start time
    - [x] Execute steps in order
    - [x] Handle plan result
    - [x] Update plan status to Completed/Failed
    - [x] Set plan completion time
    - [x] Calculate plan duration
    - [x] Update plan in storage
    - [x] Log execution
    - [x] Return plan result
  - [x] Implement ValidatePlan method
    - [x] Check if plan exists in storage
    - [x] Validate plan structure
    - [x] Validate step dependencies
    - [x] Validate step actions
    - [x] Log validation
    - [x] Return validation result
  - [x] Implement MonitorPlan method
    - [x] Check if plan exists in storage
    - [x] Check plan status
    - [x] Calculate plan progress
    - [x] Check for timeouts
    - [x] Check for errors
    - [x] Log monitoring
    - [x] Return monitoring result
  - [x] Implement AdaptPlan method
    - [x] Check if plan exists in storage
    - [x] Check if adaptation is needed
    - [x] Create adapted plan
    - [x] Update plan in storage
    - [x] Log adaptation
    - [x] Return adapted plan
  - [x] Implement RollbackStep method
    - [x] Check if plan exists in storage
    - [x] Check if step exists in plan
    - [x] Execute step rollback action
    - [x] Update step status to RolledBack
    - [x] Update plan in storage
    - [x] Log rollback
    - [x] Return rollback result
  - [x] Implement RollbackPlan method
    - [x] Check if plan exists in storage
    - [x] Rollback steps in reverse order
    - [x] Update plan status to Cancelled
    - [x] Update plan in storage
    - [x] Log rollback
    - [x] Return rollback result
  - [x] Implement GetExecutionContext method
    - [x] Check if plan exists in storage
    - [x] Return execution context if found, None otherwise
  - [x] Implement UpdateExecutionContext method
    - [x] Check if plan exists in storage
    - [x] Update execution context
    - [x] Update plan in storage
    - [x] Log update
    - [x] Return updated context

### DependencyInjection/ServiceCollectionExtensions.fs
- [x] Create ServiceCollectionExtensions module
  - [x] Implement addTarsEngineFSharpPlanning function
    - [x] Register IPlanningService
    - [x] Register ExecutionPlanner
    - [x] Return service collection

### PlanGeneration.fs
- [ ] Create PlanGeneration module
  - [ ] Implement GeneratePlanFromMetascript function
    - [ ] Parse metascript
    - [ ] Extract steps
    - [ ] Create execution plan
    - [ ] Return plan
  - [ ] Implement GeneratePlanFromImprovement function
    - [ ] Analyze improvement
    - [ ] Identify required steps
    - [ ] Create execution plan
    - [ ] Return plan
  - [ ] Implement GenerateStepFromAction function
    - [ ] Analyze action
    - [ ] Create execution step
    - [ ] Return step
  - [ ] Implement GenerateRollbackAction function
    - [ ] Analyze action
    - [ ] Create rollback action
    - [ ] Return rollback action
  - [ ] Implement GenerateValidationRules function
    - [ ] Analyze action
    - [ ] Create validation rules
    - [ ] Return rules

### PlanExecution.fs
- [ ] Create PlanExecution module
  - [ ] Implement ExecuteFileModification function
    - [ ] Backup file
    - [ ] Modify file
    - [ ] Validate modification
    - [ ] Return result
  - [ ] Implement ExecuteFileCreation function
    - [ ] Create file
    - [ ] Validate creation
    - [ ] Return result
  - [ ] Implement ExecuteFileDeletion function
    - [ ] Backup file
    - [ ] Delete file
    - [ ] Validate deletion
    - [ ] Return result
  - [ ] Implement ExecuteFileBackup function
    - [ ] Backup file
    - [ ] Validate backup
    - [ ] Return result
  - [ ] Implement ExecuteFileRestore function
    - [ ] Restore file
    - [ ] Validate restoration
    - [ ] Return result
  - [ ] Implement ExecuteValidation function
    - [ ] Validate rules
    - [ ] Return result
  - [ ] Implement ExecuteCompilation function
    - [ ] Compile code
    - [ ] Return result
  - [ ] Implement ExecuteTestExecution function
    - [ ] Execute tests
    - [ ] Return result
  - [ ] Implement ExecuteCommandExecution function
    - [ ] Execute command
    - [ ] Return result
  - [ ] Implement ExecuteApiCall function
    - [ ] Make API call
    - [ ] Return result
  - [ ] Implement ExecuteDatabaseOperation function
    - [ ] Execute database operation
    - [ ] Return result
  - [ ] Implement ExecuteNotification function
    - [ ] Send notification
    - [ ] Return result
  - [ ] Implement ExecuteApproval function
    - [ ] Request approval
    - [ ] Wait for response
    - [ ] Return result
  - [ ] Implement ExecuteDeployment function
    - [ ] Deploy changes
    - [ ] Return result
  - [ ] Implement ExecuteRollback function
    - [ ] Rollback changes
    - [ ] Return result

### PlanMonitoring.fs
- [ ] Create PlanMonitoring module
  - [ ] Implement MonitorExecution function
    - [ ] Check execution status
    - [ ] Calculate progress
    - [ ] Check for timeouts
    - [ ] Check for errors
    - [ ] Return monitoring result
  - [ ] Implement DetectAnomaly function
    - [ ] Analyze execution metrics
    - [ ] Identify anomalies
    - [ ] Return anomaly detection result
  - [ ] Implement GenerateExecutionReport function
    - [ ] Collect execution data
    - [ ] Generate report
    - [ ] Return report

### PlanAdaptation.fs
- [ ] Create PlanAdaptation module
  - [ ] Implement AdaptToFailure function
    - [ ] Analyze failure
    - [ ] Create recovery steps
    - [ ] Update plan
    - [ ] Return adapted plan
  - [ ] Implement AdaptToFeedback function
    - [ ] Analyze feedback
    - [ ] Create adaptation steps
    - [ ] Update plan
    - [ ] Return adapted plan
  - [ ] Implement AdaptToEnvironmentChange function
    - [ ] Analyze environment change
    - [ ] Create adaptation steps
    - [ ] Update plan
    - [ ] Return adapted plan
  - [ ] Implement OptimizePlan function
    - [ ] Analyze plan
    - [ ] Identify optimization opportunities
    - [ ] Create optimized plan
    - [ ] Return optimized plan

### PlanningService.fs
- [ ] Create PlanningService class
  - [ ] Implement constructor with logger and dependencies
  - [ ] Implement IPlanningService interface
  - [ ] Integrate PlanGeneration module
  - [ ] Integrate PlanExecution module
  - [ ] Integrate PlanMonitoring module
  - [ ] Integrate PlanAdaptation module

## Unit Tests

### ExecutionPlannerTests.fs
- [x] Create ExecutionPlannerTests class
  - [x] Test CreateExecutionPlan
  - [x] Test GetExecutionPlan
  - [x] Test GetAllExecutionPlans
  - [x] Test UpdateExecutionPlan
  - [x] Test DeleteExecutionPlan
  - [x] Test ExecuteStep
  - [x] Test ExecutePlan
  - [x] Test ValidatePlan
  - [x] Test MonitorPlan
  - [x] Test AdaptPlan
  - [x] Test RollbackStep
  - [x] Test RollbackPlan
  - [x] Test GetExecutionContext
  - [x] Test UpdateExecutionContext

### PlanGenerationTests.fs
- [ ] Create PlanGenerationTests class
  - [ ] Test GeneratePlanFromMetascript
  - [ ] Test GeneratePlanFromImprovement
  - [ ] Test GenerateStepFromAction
  - [ ] Test GenerateRollbackAction
  - [ ] Test GenerateValidationRules

### PlanExecutionTests.fs
- [ ] Create PlanExecutionTests class
  - [ ] Test ExecuteFileModification
  - [ ] Test ExecuteFileCreation
  - [ ] Test ExecuteFileDeletion
  - [ ] Test ExecuteFileBackup
  - [ ] Test ExecuteFileRestore
  - [ ] Test ExecuteValidation
  - [ ] Test ExecuteCompilation
  - [ ] Test ExecuteTestExecution
  - [ ] Test ExecuteCommandExecution
  - [ ] Test ExecuteApiCall
  - [ ] Test ExecuteDatabaseOperation
  - [ ] Test ExecuteNotification
  - [ ] Test ExecuteApproval
  - [ ] Test ExecuteDeployment
  - [ ] Test ExecuteRollback

### PlanMonitoringTests.fs
- [ ] Create PlanMonitoringTests class
  - [ ] Test MonitorExecution
  - [ ] Test DetectAnomaly
  - [ ] Test GenerateExecutionReport

### PlanAdaptationTests.fs
- [ ] Create PlanAdaptationTests class
  - [ ] Test AdaptToFailure
  - [ ] Test AdaptToFeedback
  - [ ] Test AdaptToEnvironmentChange
  - [ ] Test OptimizePlan

### PlanningServiceTests.fs
- [ ] Create PlanningServiceTests class
  - [ ] Test all IPlanningService methods

## Integration Tests

### PlanningIntegrationTests.fs
- [ ] Create PlanningIntegrationTests class
  - [ ] Test planning with metascript execution
  - [ ] Test planning with improvement implementation
  - [ ] Test planning with file system operations
  - [ ] Test planning with compilation and testing
  - [ ] Test planning with error handling and recovery

## Progress Notes
- 2023-06-01: Created Types.fs with all required types and helper functions
- 2023-06-05: Started implementation of IPlanningService.fs interface
- 2023-06-10: Implemented IPlanningService.fs interface
- 2023-06-15: Implemented ExecutionPlanner.fs class
- 2023-06-20: Implemented DependencyInjection/ServiceCollectionExtensions.fs
- 2023-06-25: Created unit tests for ExecutionPlanner
- 2023-06-30: Fixed compilation issues in the Planning module
