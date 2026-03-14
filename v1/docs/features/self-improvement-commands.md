# Self-Improvement Commands

TARS includes a comprehensive set of self-improvement commands that allow it to analyze, improve, generate, and test code. These commands are available through the `self-improve` command in the TARS CLI.

## Available Commands

### Analyze

Analyzes code for potential improvements.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve analyze path/to/file.cs
```

Options:
- `--project, -p`: Path to the project (if analyzing a single file)
- `--recursive, -r`: Analyze recursively (for directories)
- `--max-files, -m`: Maximum number of files to analyze (default: 10)

### Improve

Improves code based on analysis.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve improve path/to/file.cs
```

Options:
- `--project, -p`: Path to the project (if improving a single file)
- `--recursive, -r`: Improve recursively (for directories)
- `--max-files, -m`: Maximum number of files to improve (default: 5)
- `--backup, -b`: Create backups of original files (default: true)

### Generate

Generates code based on requirements.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve generate path/to/output.cs --requirements "Create a simple calculator class"
```

Options:
- `--project, -p`: Path to the project
- `--requirements, -r`: Requirements for the code
- `--language, -l`: Programming language

### Test

Generates and runs tests for a file.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve test path/to/file.cs
```

Options:
- `--project, -p`: Path to the project (if testing a single file)
- `--output, -o`: Path to the output test file

### Cycle

Runs a complete self-improvement cycle on a project.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve cycle path/to/project
```

Options:
- `--max-files, -m`: Maximum number of files to improve (default: 10)
- `--backup, -b`: Create backups of original files (default: true)
- `--test, -t`: Run tests after improvements (default: true)

### Feedback

Records feedback on code generation or improvement.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve feedback path/to/file.cs --rating 5 --comment "Great improvement!"
```

Options:
- `--rating, -r`: Rating (1-5)
- `--comment, -c`: Comment
- `--type, -t`: Feedback type (Generation, Improvement, Test)

### Stats

Shows learning statistics.

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve stats
```

## Examples

### Analyzing a File

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve analyze Examples/TestFile.cs
```

This command will analyze the `TestFile.cs` file and suggest improvements.

### Generating Code

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve generate Examples/Calculator.cs --requirements "Create a simple calculator class with add, subtract, multiply, and divide methods"
```

This command will generate a calculator class based on the requirements.

### Generating Tests

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve test Examples/Calculator.cs --output Examples/CalculatorTests.cs
```

This command will generate tests for the `Calculator.cs` file and save them to `CalculatorTests.cs`.

### Running a Self-Improvement Cycle

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-improve cycle Examples --max-files 5 --backup
```

This command will run a complete self-improvement cycle on the `Examples` directory, analyzing and improving up to 5 files, and creating backups of the original files.

## Implementation Details

The self-improvement commands are implemented in the `SelfImprovementController` class in the `TarsCli` project. The controller uses the following services:

- `SelfImprovementService`: Coordinates the self-improvement process
- `CodeAnalysisService`: Analyzes code for potential improvements
- `ProjectAnalysisService`: Analyzes project structure
- `CodeGenerationService`: Generates code based on requirements
- `CodeExecutionService`: Executes code and tests
- `LearningService`: Tracks learning progress and feedback

These services are implemented in the `TarsEngine` project and are designed to be extensible and reusable.

## Future Enhancements

- **Improved Pattern Recognition**: Enhance the ability to recognize patterns in code
- **More Sophisticated Improvements**: Implement more sophisticated improvement suggestions
- **Better Learning**: Improve the learning system to better track and learn from feedback
- **Integration with CI/CD**: Integrate self-improvement with CI/CD pipelines
- **Support for More Languages**: Add support for more programming languages
