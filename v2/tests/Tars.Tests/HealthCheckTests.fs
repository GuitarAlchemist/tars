module Tars.Tests.HealthCheckTests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core

[<Fact>]
let ``HealthCheckRegistry runs all checks`` () =
    task {
        // Arrange
        let registry = HealthCheckRegistry()
        registry.RegisterSimple("check1", [ "test" ], fun () -> Task.FromResult(Healthy))
        registry.RegisterSimple("check2", [ "test" ], fun () -> Task.FromResult(Healthy))

        // Act
        let! report = registry.RunAllAsync()

        // Assert
        Assert.Equal(2, report.Checks.Length)
        Assert.Equal(Healthy, report.Status)
    }

[<Fact>]
let ``HealthCheckRegistry detects unhealthy check`` () =
    task {
        // Arrange
        let registry = HealthCheckRegistry()
        registry.RegisterSimple("healthy", [ "test" ], fun () -> Task.FromResult(Healthy))
        registry.RegisterSimple("unhealthy", [ "test" ], fun () -> Task.FromResult(Unhealthy "Failed"))

        // Act
        let! report = registry.RunAllAsync()

        // Assert
        match report.Status with
        | Unhealthy _ -> Assert.True(true)
        | _ -> Assert.Fail("Expected Unhealthy status")
    }

[<Fact>]
let ``HealthCheckRegistry detects degraded check`` () =
    task {
        // Arrange
        let registry = HealthCheckRegistry()
        registry.RegisterSimple("healthy", [ "test" ], fun () -> Task.FromResult(Healthy))
        registry.RegisterSimple("degraded", [ "test" ], fun () -> Task.FromResult(Degraded "Warning"))

        // Act
        let! report = registry.RunAllAsync()

        // Assert
        match report.Status with
        | Degraded _ -> Assert.True(true)
        | _ -> Assert.Fail("Expected Degraded status")
    }

[<Fact>]
let ``HealthCheckRegistry filters by tag`` () =
    task {
        // Arrange
        let registry = HealthCheckRegistry()
        registry.RegisterSimple("tagged", [ "special" ], fun () -> Task.FromResult(Healthy))
        registry.RegisterSimple("untagged", [ "other" ], fun () -> Task.FromResult(Healthy))

        // Act
        let! report = registry.RunByTagAsync("special")

        // Assert
        Assert.Single(report.Checks) |> ignore
        Assert.Equal("tagged", report.Checks.Head.Name)
    }

[<Fact>]
let ``HealthCheck handles exceptions`` () =
    task {
        // Arrange
        let registry = HealthCheckRegistry()

        registry.RegisterSimple(
            "failing",
            [ "test" ],
            fun () -> Task.FromException<HealthStatus>(Exception("Test error"))
        )

        // Act
        let! report = registry.RunAllAsync()

        // Assert
        match report.Checks.Head.Status with
        | Unhealthy msg -> Assert.Contains("Exception", msg)
        | _ -> Assert.Fail("Expected Unhealthy with exception")
    }

[<Fact>]
let ``alwaysHealthy returns Healthy`` () =
    task {
        // Arrange
        let check = HealthChecks.alwaysHealthy "test"

        // Act
        let! status = (check :> IHealthCheck).CheckAsync()

        // Assert
        Assert.Equal(Healthy, status)
    }

[<Fact>]
let ``memoryCheck returns status based on threshold`` () =
    task {
        // Arrange - use very high threshold
        let check = HealthChecks.memoryCheck 10000.0

        // Act
        let! status = (check :> IHealthCheck).CheckAsync()

        // Assert - should be healthy with high threshold
        Assert.Equal(Healthy, status)
    }

[<Fact>]
let ``formatReport produces output`` () =
    // Arrange
    let report =
        { Status = Healthy
          TotalDuration = TimeSpan.FromMilliseconds(100.0)
          Checks =
            [ { Name = "test"
                Status = Healthy
                Duration = TimeSpan.FromMilliseconds(50.0)
                Tags = [ "test" ]
                Data = Map.empty } ]
          Timestamp = DateTime.UtcNow }

    // Act
    let output = HealthChecks.formatReport report

    // Assert
    Assert.Contains("Health Report", output)
    Assert.Contains("Healthy", output)
    Assert.Contains("test", output)

[<Fact>]
let ``formatJson produces valid output`` () =
    // Arrange
    let report =
        { Status = Healthy
          TotalDuration = TimeSpan.FromMilliseconds(100.0)
          Checks =
            [ { Name = "test"
                Status = Healthy
                Duration = TimeSpan.FromMilliseconds(50.0)
                Tags = [ "test" ]
                Data = Map.empty } ]
          Timestamp = DateTime.UtcNow }

    // Act
    let json = HealthChecks.formatJson report

    // Assert
    Assert.Contains("\"status\":\"healthy\"", json)
    Assert.Contains("\"name\":\"test\"", json)
