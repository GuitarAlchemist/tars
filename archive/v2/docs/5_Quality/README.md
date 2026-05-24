# Quality Assurance

Testing strategies, metrics, and quality guidelines for TARS v2.

## Testing

### Quick Reference

- **[Testing Tips](./Testing_Tips.md)**: Commands, patterns, and best practices
- **Run all tests**: `dotnet test`
- **Watch mode**: `dotnet watch test`

### Test Coverage

- **Target**: Minimum 80% coverage for core components
- **Current**: *(Run coverage report to update)*

## Metrics & Evaluation

- **[Competence Metrics](./Competence_Metrics.md)**: Runtime performance and quality metrics  
- **Offline Evaluation**: Benchmarking against standard datasets
- **Trace Analysis**: Debugging and performance profiling

## Grammar & Prompt Engineering

- **[Grammar Distillation](./Grammar_Distillation.md)**: Structured output scaffolding
- **Prompt Patterns**: Best practices for LLM interactions

## Quality Standards

### Code Quality

- **Zero Tolerance Policy**: No placeholders or fake implementations
- **F#-First**: New development in F#, C# for infrastructure only
- **Conventional Commits**: Use conventional commit format

### Test Quality  

- **Arrange-Act-Assert**: Clear test structure
- **Isolated Tests**: No shared mutable state
- **Fast Feedback**: Unit tests complete in <5s

## CI/CD

*(TODO: Add CI/CD pipeline documentation)*
