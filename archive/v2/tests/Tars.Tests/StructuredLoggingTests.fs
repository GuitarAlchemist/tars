module Tars.Tests.StructuredLoggingTests

open System
open Xunit
open Tars.Core

[<Fact>]
let ``LoggingConfig Default has correct values`` () =
    let config = LoggingConfig.Default

    Assert.Equal(LogLevel.Info, config.MinLevel)
    Assert.True(config.OutputJson)
    Assert.True(config.IncludeTimestamp)

[<Fact>]
let ``LoggingConfig Development has debug level`` () =
    let config = LoggingConfig.Development

    Assert.Equal(LogLevel.Debug, config.MinLevel)
    Assert.False(config.OutputJson)

[<Fact>]
let ``StructuredLogger can be created with category`` () =
    let config = LoggingConfig.Default
    let logger = StructuredLogger(config, "TestCategory")

    // Just verify it creates without error
    Assert.NotNull(logger)

[<Fact>]
let ``StructuredLogger WithCorrelation returns new logger`` () =
    let config = LoggingConfig.Default
    let logger = StructuredLogger(config) :> IStructuredLogger

    let correlationId = Guid.NewGuid()
    let scopedLogger = logger.WithCorrelation(correlationId)

    Assert.NotNull(scopedLogger)

[<Fact>]
let ``StructuredLogger WithCategory returns new logger`` () =
    let config = LoggingConfig.Default
    let logger = StructuredLogger(config) :> IStructuredLogger

    let scopedLogger = logger.WithCategory("Custom")

    Assert.NotNull(scopedLogger)

[<Fact>]
let ``StructuredLogger WithProperty returns new logger`` () =
    let config = LoggingConfig.Default
    let logger = StructuredLogger(config) :> IStructuredLogger

    let scopedLogger = logger.WithProperty("key", "value")

    Assert.NotNull(scopedLogger)

[<Fact>]
let ``Logging module initializes default logger`` () =
    // Act
    Logging.init LoggingConfig.Default
    let logger = Logging.logger ()

    // Assert
    Assert.NotNull(logger)

[<Fact>]
let ``Logging withCorrelation creates scoped logger`` () =
    // Arrange
    Logging.init LoggingConfig.Default

    // Act
    let scoped = Logging.withCorrelation (Guid.NewGuid())

    // Assert
    Assert.NotNull(scoped)

[<Fact>]
let ``Logging withCategory creates scoped logger`` () =
    // Arrange
    Logging.init LoggingConfig.Default

    // Act
    let scoped = Logging.withCategory "TestCategory"

    // Assert
    Assert.NotNull(scoped)

[<Fact>]
let ``LogEntry record has correct structure`` () =
    let entry =
        { Timestamp = DateTime.UtcNow
          Level = LogLevel.Info
          Message = "Test message"
          Category = "Test"
          CorrelationId = Some(Guid.NewGuid())
          TraceId = Some "trace-123"
          SpanId = Some "span-456"
          Properties = Map.ofList [ ("key", box "value") ]
          Exception = None }

    Assert.Equal(LogLevel.Info, entry.Level)
    Assert.Equal("Test message", entry.Message)
    Assert.True(entry.CorrelationId.IsSome)
    Assert.True(entry.Properties.Count > 0)
