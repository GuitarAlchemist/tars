# Project Guidelines

## 1. Build/Configuration Instructions

### Prerequisites
- .NET 10.0 SDK (Preview or RC depending on availability for `net10.0`)
- Docker (required for `Tars.Sandbox`)

### Building the Project
To build the entire solution, run the following command from the repository root:

```bash
dotnet build
```

This will restore dependencies and build all projects in the solution.

## 2. Testing Information

The project uses xUnit for testing.

### Running Tests
To execute all tests in the solution:

```bash
dotnet test
```

### Adding New Tests
To add a new test project:
1. Create a new xUnit project:
   ```bash
   dotnet new xunit -lang F# -n YourProject.Tests -o tests/YourProject.Tests
   ```
2. Add the project to the solution:
   ```bash
   dotnet sln add tests/YourProject.Tests/YourProject.Tests.fsproj
   ```
3. Add references to the projects you want to test:
   ```bash
   dotnet add tests/YourProject.Tests/YourProject.Tests.fsproj reference src/Tars.Core/Tars.Core.fsproj
   ```

### Example Test
Here is a simple example of an F# xUnit test (`Tests.fs`):

```fsharp
module Tests

open System
open Xunit

[<Fact>]
let ``My test`` () =
    Assert.True(true)
```

## 3. Additional Development Information

- **Language**: F#
- **Framework**: .NET 10.0 (`net10.0`)
- **Code Style**: Follow standard F# coding conventions.
- **Key Libraries**:
    - `Docker.DotNet`: Used in `Tars.Sandbox` for container management.
- **Environment**:
    - `Tars.Sandbox` adapts to Windows (Named Pipes) and Linux/Unix (Unix Sockets) for Docker connection.
