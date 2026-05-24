namespace TarsTaskManager.Core.Domain

open System

/// Task priority levels
type Priority = 
    | Low = 1
    | Medium = 2  
    | High = 3
    | Critical = 4

/// Task status enumeration
type TaskStatus =
    | Pending
    | InProgress
    | Completed
    | Cancelled
    | OnHold

/// User domain model
type User = {
    Id: Guid
    Email: string
    FirstName: string
    LastName: string
    AvatarUrl: string option
    Timezone: string
    CreatedAt: DateTime
    IsActive: bool
}

/// Task domain model
type Task = {
    Id: Guid
    Title: string
    Description: string option
    Priority: Priority
    Status: TaskStatus
    DueDate: DateTime option
    EstimatedDuration: TimeSpan option
    Tags: string list
    CreatedBy: Guid
    AssignedTo: Guid option
    ProjectId: Guid option
    CreatedAt: DateTime
    UpdatedAt: DateTime
}

/// Project domain model
type Project = {
    Id: Guid
    Name: string
    Description: string option
    Color: string
    OwnerId: Guid
    Status: string
    CreatedAt: DateTime
}

/// AI-powered task analysis result
type TaskAnalysis = {
    SuggestedPriority: Priority
    EstimatedDuration: TimeSpan
    SuggestedTags: string list
    PredictedDueDate: DateTime option
    Confidence: float
}
