# Integration Testing Guide for Autonomous Execution System

This guide provides instructions for testing the Autonomous Execution System end-to-end.

## Prerequisites

Before running the integration tests, ensure you have:

1. A development environment with TARS installed
2. Access to the TARS codebase
3. Sufficient permissions to execute commands and modify files

## Test Scenarios

### Scenario 1: Basic Execution Flow

This scenario tests the basic execution flow of the Autonomous Execution System.

#### Steps

1. Create a test improvement:
   ```bash
   tarscli improve create --name "Test Improvement" --description "A test improvement for integration testing" --category "Test"
   ```

2. Execute the improvement in dry run mode:
   ```bash
   tarscli execute improvement --id <improvement-id> --dry-run
   ```

3. Monitor the execution:
   ```bash
   tarscli execute monitor --context-id <context-id>
   ```

4. Review the execution status:
   ```bash
   tarscli execute status --context-id <context-id> --detail
   ```

#### Expected Results

- The improvement should be executed in dry run mode without making actual changes
- The execution status should show the improvement as completed
- The execution logs should show the steps that would be executed

### Scenario 2: Execution with Validation

This scenario tests the execution flow with validation.

#### Steps

1. Create a test improvement that modifies a file:
   ```bash
   tarscli improve create --name "Test Improvement with Validation" --description "A test improvement that modifies a file" --category "Test"
   ```

2. Execute the improvement with validation:
   ```bash
   tarscli execute improvement --id <improvement-id> --validate
   ```

3. Monitor the execution:
   ```bash
   tarscli execute monitor --context-id <context-id>
   ```

4. Review the execution status:
   ```bash
   tarscli execute status --context-id <context-id> --detail
   ```

#### Expected Results

- The improvement should be executed with validation
- The validation should check syntax, semantics, and run tests
- The execution status should show the validation results
- If validation passes, the changes should be applied

### Scenario 3: Execution with Rollback

This scenario tests the execution flow with rollback.

#### Steps

1. Create a test improvement that modifies a file:
   ```bash
   tarscli improve create --name "Test Improvement with Rollback" --description "A test improvement that modifies a file" --category "Test"
   ```

2. Execute the improvement with auto-rollback:
   ```bash
   tarscli execute improvement --id <improvement-id> --auto-rollback
   ```

3. Introduce a validation failure (e.g., by modifying the file to contain syntax errors)

4. Monitor the execution:
   ```bash
   tarscli execute monitor --context-id <context-id>
   ```

5. Review the execution status:
   ```bash
   tarscli execute status --context-id <context-id> --detail
   ```

#### Expected Results

- The improvement should be executed with auto-rollback
- The validation should fail due to the introduced errors
- The changes should be automatically rolled back
- The execution status should show the rollback as completed

### Scenario 4: Manual Review and Approval

This scenario tests the manual review and approval flow.

#### Steps

1. Create a test improvement that modifies a file:
   ```bash
   tarscli improve create --name "Test Improvement with Review" --description "A test improvement that requires review" --category "Test"
   ```

2. Execute the improvement:
   ```bash
   tarscli execute improvement --id <improvement-id>
   ```

3. Monitor the execution:
   ```bash
   tarscli execute monitor --context-id <context-id>
   ```

4. Review the changes:
   ```bash
   tarscli execute review --context-id <context-id>
   ```

5. Approve the changes:
   ```bash
   tarscli execute review --context-id <context-id> --approve --comment "Approved after review"
   ```

6. Review the execution status:
   ```bash
   tarscli execute status --context-id <context-id> --detail
   ```

#### Expected Results

- The improvement should be executed and await review
- The review command should show the changes
- After approval, the changes should be committed
- The execution status should show the approval and completion

### Scenario 5: Manual Rollback

This scenario tests the manual rollback flow.

#### Steps

1. Create a test improvement that modifies a file:
   ```bash
   tarscli improve create --name "Test Improvement with Manual Rollback" --description "A test improvement that will be manually rolled back" --category "Test"
   ```

2. Execute the improvement:
   ```bash
   tarscli execute improvement --id <improvement-id>
   ```

3. Monitor the execution:
   ```bash
   tarscli execute monitor --context-id <context-id>
   ```

4. Review the changes:
   ```bash
   tarscli execute review --context-id <context-id>
   ```

5. Roll back the changes:
   ```bash
   tarscli execute rollback --context-id <context-id> --all
   ```

6. Review the execution status:
   ```bash
   tarscli execute status --context-id <context-id> --detail
   ```

#### Expected Results

- The improvement should be executed successfully
- The rollback command should revert all changes
- The execution status should show the rollback as completed
- The files should be restored to their original state

## Test Matrix

| Scenario | Dry Run | Validation | Auto-Rollback | Manual Review | Manual Rollback |
|----------|---------|------------|---------------|---------------|-----------------|
| 1        | Yes     | No         | No            | No            | No              |
| 2        | No      | Yes        | No            | No            | No              |
| 3        | No      | Yes        | Yes           | No            | No              |
| 4        | No      | Yes        | No            | Yes           | No              |
| 5        | No      | Yes        | No            | Yes           | Yes             |

## Troubleshooting

### Common Issues

#### Execution Plan Creation Fails

- Check if the improvement exists
- Verify that the improvement has valid content
- Ensure the improvement is in a valid state

#### Validation Fails

- Check for syntax errors in the code
- Look for semantic issues like undefined variables
- Ensure tests are passing

#### Rollback Fails

- Check if backup files exist
- Verify that the rollback context is valid
- Ensure the file system is accessible

### Logging

To get detailed logs for troubleshooting, use the `--detail` option with the `execute status` command:

```bash
tarscli execute status --context-id <context-id> --detail
```

## Reporting Issues

If you encounter issues during integration testing, please report them with the following information:

1. Scenario being tested
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Logs and error messages
6. Environment information

## Conclusion

This integration testing guide provides a comprehensive set of scenarios to test the Autonomous Execution System end-to-end. By following these scenarios, you can ensure that the system works correctly and reliably.
