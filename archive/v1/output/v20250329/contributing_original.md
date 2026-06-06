# Contributing to TARS

Thank you for your interest in contributing to TARS! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

TARS is committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to adhere to our Code of Conduct, which promotes respect, empathy, and collaboration.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **.NET 9 SDK** or later
- **Git** for version control
- **Ollama** for local language model inference
- **PowerShell** (Windows) or **Bash** (Linux/macOS)

### Setting Up the Development Environment

1. **Fork the Repository**

   Start by forking the [TARS repository](https://github.com/GuitarAlchemist/tars) on GitHub.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/YOUR-USERNAME/tars.git
   cd tars
   ```

3. **Add the Upstream Remote**

   ```bash
   git remote add upstream https://github.com/GuitarAlchemist/tars.git
   ```

4. **Install Prerequisites**

   ```bash
   # Windows
   .\Scripts\Install-Prerequisites.ps1

   # Linux/macOS
   ./Scripts/install-prerequisites.sh
   ```

5. **Build the Project**

   ```bash
   dotnet build
   ```

6. **Run the Tests**

   ```bash
   dotnet test
   ```

## Development Workflow

### Branching Strategy

We use a simplified Git workflow:

- `main`: The main branch containing stable code
- Feature branches: Created for new features or bug fixes

### Creating a Feature Branch

1. Ensure your fork is up to date:

   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

2. Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

   Use a descriptive name for your branch, prefixed with `feature/` for features or `fix/` for bug fixes.

### Making Changes

1. Make your changes to the codebase.
2. Commit your changes with a descriptive commit message:

   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

3. Push your changes to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

1. **Create a Pull Request**

   Go to the [TARS repository](https://github.com/GuitarAlchemist/tars) and create a new pull request from your feature branch.

2. **Describe Your Changes**

   Provide a clear description of your changes, including:
   - What problem does this solve?
   - How does it solve the problem?
   - Any breaking changes?
   - Screenshots or examples (if applicable)

3. **Review Process**

   - A maintainer will review your pull request
   - Address any feedback or requested changes
   - Once approved, your pull request will be merged

4. **After Merge**

   After your pull request is merged, you can:
   - Delete your feature branch
   - Pull the changes to your local main branch
   - Start working on a new feature

## Coding Standards

### C# Coding Standards

- Follow the [Microsoft C# Coding Conventions](https://docs.microsoft.com/en-us/dotnet/csharp/fundamentals/coding-style/coding-conventions)
- Use meaningful names for variables, methods, and classes
- Write clear comments for complex logic
- Keep methods focused and concise
- Use proper exception handling

### F# Coding Standards

- Follow the [F# Component Design Guidelines](https://docs.microsoft.com/en-us/dotnet/fsharp/style-guide/component-design-guidelines)
- Prefer immutable data structures
- Use pattern matching effectively
- Write clear type signatures
- Leverage the F# type system for safety

### General Guidelines

- Write self-documenting code
- Follow the SOLID principles
- Keep dependencies minimal and explicit
- Write unit tests for new functionality
- Document public APIs

## Testing

### Writing Tests

- Write unit tests for all new functionality
- Use xUnit for testing
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Test edge cases and error conditions

### Running Tests

```bash
# Run all tests
dotnet test

# Run tests for a specific project
dotnet test TarsEngine.Tests/TarsEngine.Tests.fsproj

# Run tests with a specific filter
dotnet test --filter "Category=UnitTest"
```

## Documentation

### Code Documentation

- Document all public APIs with XML comments
- Explain the purpose of classes, methods, and properties
- Document parameters, return values, and exceptions
- Provide examples for complex functionality

### Project Documentation

- Update relevant documentation for new features
- Create new documentation files as needed
- Follow the existing documentation structure
- Use clear, concise language

## Issue Reporting

### Bug Reports

When reporting a bug, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, .NET version, etc.)
6. Screenshots or error messages (if applicable)

### Feature Requests

When requesting a feature, please include:

1. A clear, descriptive title
2. A detailed description of the feature
3. The problem it solves
4. Any alternatives you've considered
5. Examples or mockups (if applicable)

## Feature Requests

We welcome feature requests! To request a new feature:

1. Check if the feature has already been requested
2. Create a new issue with the "Feature Request" template
3. Provide a detailed description of the feature
4. Explain why it would be valuable
5. Consider implementing it yourself and submitting a pull request

## Community

### Communication Channels

- **GitHub Issues**: For bug reports, feature requests, and general discussion
- **GitHub Discussions**: For questions, ideas, and community interaction
- **Pull Requests**: For code contributions and reviews

### Recognition

All contributors will be recognized in the project's contributors list. We value every contribution, whether it's code, documentation, testing, or feedback.

## Thank You!

Thank you for contributing to TARS! Your efforts help make the project better for everyone. We appreciate your time and dedication.
