# Contributing to TARS

Thank you for your interest in contributing to TARS! This guide will help you get started with contributing to the project.

## Table of Contents

* [Code of Conduct](#code-of-conduct)
* [Getting Started](#getting-started)
* [Development Workflow](#development-workflow)
* [Pull Request Process](#pull-request-process)
* [Coding Standards](#coding-standards)
* [Testing](#testing)
* [Documentation](#documentation)
* [Issue Reporting](#issue-reporting)
* [Feature Requests](#feature-requests)
* [Community](#community)

## Code of Conduct

TARS is committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to adhere to our Code of Conduct, which promotes respect, empathy, and collaboration.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

* `.NET 9 SDK` or later
* `Git` for version control
* `Ollama` for local language model inference
* `PowerShell` (Windows) or `Bash` (Linux/macOS)

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
   .\Scripts\install-prerequisites.ps1

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

* `main`: The main branch containing stable code
* Feature branches: Created for new features or bug fixes

### Creating a Feature Branch

1. Create a new branch based on the latest `main` branch.
2. Make changes and commit them to your local repository.
3. Push your changes to GitHub.

### Merging Changes

1. Pull the latest `main` branch from GitHub.
2. Merge your feature branch into `main`.
3. Push your changes to GitHub.

## Pull Request Process

When submitting a pull request:

* Ensure that your code adheres to our coding standards and best practices.
* Provide clear, concise comments explaining your code.
* Test your changes thoroughly before submitting the pull request.
* Be prepared to address any questions or concerns from reviewers.

## Coding Standards

We follow the .NET Core guidelines for coding standards. Please familiarize yourself with these guidelines before making any contributions.

## Testing

### Writing Tests

* Write unit tests for all new functionality using xUnit.
* Use the Arrange-Act-Assert pattern.
* Mock external dependencies as needed.
* Test edge cases and error conditions.

### Running Tests