# TARS Migration Guide

This guide provides instructions for migrating from the old model classes to the unified model classes.

## Overview

We are consolidating model classes from multiple projects into a single unified project to reduce duplication and improve maintainability. The following projects are being consolidated:

- TarsEngine.Models
- TarsEngine.Models.Core

The unified model classes are now in:

- TarsEngine.Models.Unified
- TarsEngine.Unified

## Migration Steps

### 1. Update Project References

Remove references to:
- TarsEngine.Models.Core

Add references to:
- TarsEngine.Models.Unified
- TarsEngine.Unified

### 2. Update Using Statements

Add the following using statements to your files:

```csharp
using TarsEngine.Models.Unified.CodeAnalysis;
using TarsEngine.Models.Unified.Validation;
using TarsEngine.Models.Unified.Analysis;
using TarsEngine.Models.Unified.Implementation;
using TarsEngine.Models.Unified.Transactions;
using TarsEngine.Models.Unified.Metascript;
using TarsEngine.Models.Unified.Testing;
using TarsEngine.Unified.Execution;
using TarsEngine.Unified.VirtualFileSystem;
using TarsEngine.Unified.FileSystem;
using TarsEngine.Unified.Models;
```

### 3. Update Type References

Update your code to use the unified types:

| Old Type | Unified Type |
|----------|--------------|
| TarsEngine.Models.CodeIssue | TarsEngine.Models.Unified.CodeAnalysis.CodeIssue |
| TarsEngine.Models.IssueSeverity | TarsEngine.Models.Unified.CodeAnalysis.CodeIssueSeverity |
| TarsEngine.Models.CodeIssueType | TarsEngine.Models.Unified.CodeAnalysis.CodeIssueType |
| TarsEngine.Models.ValidationResult | TarsEngine.Models.Unified.Validation.ValidationResult |
| TarsEngine.Models.ExecutionMode | TarsEngine.Unified.Execution.ExecutionMode |
| TarsEngine.Models.ExecutionEnvironment | TarsEngine.Unified.Execution.ExecutionEnvironment |
| TarsEngine.Models.RelationshipType | TarsEngine.Unified.Models.RelationshipType |
| TarsEngine.Models.VirtualFileSystemContext | TarsEngine.Unified.VirtualFileSystem.VirtualFileSystemContext |
| TarsEngine.Models.ProjectAnalysisResult | TarsEngine.Models.Unified.Analysis.ProjectAnalysisResult |
| TarsEngine.Models.SolutionAnalysisResult | TarsEngine.Models.Unified.Analysis.SolutionAnalysisResult |
| TarsEngine.Models.ImplementationPlan | TarsEngine.Models.Unified.Implementation.ImplementationPlan |
| TarsEngine.Models.ImplementationStep | TarsEngine.Models.Unified.Implementation.ImplementationStep |
| TarsEngine.Models.AffectedComponent | TarsEngine.Models.Unified.Implementation.AffectedComponent |
| TarsEngine.Models.TaskComplexity | TarsEngine.Models.Unified.Implementation.TaskComplexity |
| TarsEngine.Models.Transaction | TarsEngine.Models.Unified.Transactions.Transaction |
| TarsEngine.Models.OperationType | TarsEngine.Models.Unified.Transactions.OperationType |
| TarsEngine.Models.MetascriptTemplate | TarsEngine.Models.Unified.Metascript.MetascriptTemplate |
| TarsEngine.Models.MetascriptParameter | TarsEngine.Models.Unified.Metascript.MetascriptParameter |
| TarsEngine.Models.PatternMatch | TarsEngine.Models.Unified.Metascript.PatternMatch |
| TarsEngine.Models.TestFailure | TarsEngine.Models.Unified.Testing.TestFailure |
| TarsEngine.Models.CodeStructure | TarsEngine.Models.Unified.CodeAnalysis.CodeStructure |
| TarsEngine.Models.ExecutionPermissions | TarsEngine.Unified.Execution.ExecutionPermissions |
| TarsEngine.Models.FileOperation | TarsEngine.Unified.FileSystem.FileOperation |

### 4. Use Extension Methods for Conversion

If you need to convert between old and unified types, use the extension methods in `TarsEngine.Models.Unified.Extensions.StandardModelExtensions`:

```csharp
// Convert from TarsEngine.Models.CodeIssue to TarsEngine.Models.Unified.CodeAnalysis.CodeIssue
var unifiedIssue = StandardModelExtensions.ToUnifiedFromStandard(oldIssue);

// Convert from TarsEngine.Models.ValidationResult to TarsEngine.Models.Unified.Validation.ValidationResult
var unifiedResult = StandardModelExtensions.ToUnifiedFromStandard(oldResult, true);
```

### 5. Use Fully Qualified Type Names for Ambiguous Types

If you encounter ambiguous type references, use fully qualified type names:

```csharp
// Instead of:
var issue = new CodeIssue();

// Use:
var issue = new TarsEngine.Models.Unified.CodeAnalysis.CodeIssue();
```

### 6. Create Type Aliases for Frequently Used Types

You can create type aliases to make your code more readable:

```csharp
using UnifiedCodeIssue = TarsEngine.Models.Unified.CodeAnalysis.CodeIssue;
using UnifiedValidationResult = TarsEngine.Models.Unified.Validation.ValidationResult;

// Then use:
var issue = new UnifiedCodeIssue();
var result = new UnifiedValidationResult();
```

## Troubleshooting

If you encounter errors like:

```
error CS0433: The type 'CodeIssue' exists in both 'TarsEngine.Models.Core, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null' and 'TarsEngine.Models, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null'
```

This means you still have references to both TarsEngine.Models.Core and TarsEngine.Models. Update your code to use the unified types and remove the reference to TarsEngine.Models.Core.

## Need Help?

If you need help with the migration, please contact the TARS team.
