# TARS DSL Telemetry Implementation

## Overview

The TARS DSL telemetry system collects data about parser usage, performance, and errors to help improve the parser over time. The system is designed to be:

- **Privacy-focused**: User data is anonymized by default
- **Configurable**: Users can enable/disable telemetry collection
- **Comprehensive**: Collects detailed metrics about parsing performance
- **Lightweight**: Minimal impact on parsing performance

## Architecture

The telemetry system consists of several components:

1. **TelemetryData**: Defines the data structures for telemetry
2. **TelemetryCollector**: Collects telemetry data during parsing
3. **TelemetryStorage**: Stores telemetry data to disk
4. **TelemetryReporter**: Generates reports from telemetry data
5. **TelemetryPrivacy**: Anonymizes telemetry data
6. **TelemetryConfiguration**: Configures telemetry collection
7. **TelemetryService**: Integrates telemetry collection with parsers

## Data Collection

The telemetry system collects three types of data:

### 1. Parser Usage Telemetry

- Parser type used (Original, FParsec, Incremental)
- File size and line count
- Block count, property count, and nested block count
- Parse time

### 2. Parsing Performance Telemetry

- Tokenizing time
- Block parsing time
- Property parsing time
- Nested block parsing time
- Chunking time (for incremental parser)
- Chunk parsing time (for incremental parser)
- Chunk combining time (for incremental parser)
- Chunk count and cached chunk count
- Peak memory usage

### 3. Error and Warning Telemetry

- Error count
- Warning count
- Information count
- Hint count
- Suppressed warning count

## Enhanced Telemetry Implementation

The enhanced telemetry implementation adds several improvements:

### 1. Detailed Performance Metrics

The enhanced implementation adds functions to measure specific parts of the parsing process:

```fsharp
// Start measuring a performance metric
let tokenizingMetric = TelemetryService.startMeasuring "Tokenizing"

// Perform tokenizing
// ...

// Stop measuring the performance metric
TelemetryService.stopMeasuring tokenizingMetric
```

### 2. Error and Warning Collection

The enhanced implementation adds functions to record diagnostics and suppressed warnings:

```fsharp
// Record diagnostics
TelemetryService.recordDiagnostics "EnhancedParser" diagnostics

// Record suppressed warnings
TelemetryService.recordSuppressedWarnings "EnhancedParser" suppressedWarnings
```

### 3. Memory Usage Tracking

The enhanced implementation adds functions to track memory usage:

```fsharp
// Record memory usage
TelemetryService.recordMetric "PeakMemoryUsage" (TelemetryService.getCurrentMemoryUsage())
```

## Integration with Parsers

The telemetry system is integrated with the parsers through the `TelemetryService` module. The enhanced implementation provides a more detailed integration example in `ParserWithEnhancedTelemetry.fs`.

### Basic Integration

```fsharp
let parse (code: string) =
    if TelemetryService.isTelemetryEnabled() then
        TelemetryService.parseWithTelemetry "ParserName" parseInternal code
    else
        parseInternal code
```

### Enhanced Integration

```fsharp
let parse (code: string) =
    if not (TelemetryService.isTelemetryEnabled()) then
        parseInternal code
    else
        // Start measuring specific parts of the parsing process
        let tokenizingMetric = TelemetryService.startMeasuring "Tokenizing"
        // Tokenize the code
        TelemetryService.stopMeasuring tokenizingMetric
        
        let blockParsingMetric = TelemetryService.startMeasuring "BlockParsing"
        // Parse blocks
        TelemetryService.stopMeasuring blockParsingMetric
        
        // Record memory usage
        TelemetryService.recordMetric "PeakMemoryUsage" (TelemetryService.getCurrentMemoryUsage())
        
        // Record diagnostics and suppressed warnings
        TelemetryService.recordDiagnostics "ParserName" diagnostics
        TelemetryService.recordSuppressedWarnings "ParserName" suppressedWarnings
        
        // Use the telemetry service to collect and store telemetry data
        TelemetryService.parseWithTelemetry "ParserName" parseInternal code
```

## Testing

The telemetry system includes comprehensive tests in `TestTelemetry.fs` and `TestEnhancedTelemetry.fs`. These tests verify that:

- Telemetry data is collected correctly
- Telemetry data is stored correctly
- Telemetry data is anonymized correctly
- Telemetry reports are generated correctly

## Configuration

Users can configure telemetry collection through the `TelemetryConfiguration` module:

```fsharp
// Enable telemetry collection
TelemetryService.enableTelemetry()

// Disable telemetry collection
TelemetryService.disableTelemetry()

// Configure telemetry collection
TelemetryConfiguration.configureTelemetryCollection 
    true  // Collect usage telemetry
    true  // Collect performance telemetry
    true  // Collect error and warning telemetry
    "path/to/config.json"
```

## Privacy

The telemetry system anonymizes data by default. The `TelemetryPrivacy` module provides functions to:

- Anonymize telemetry data
- Purge telemetry data

## Reporting

The telemetry system generates reports from collected data. The `TelemetryReporter` module provides functions to:

- Generate summary reports
- Generate detailed reports
- Save reports to files

## Next Steps

1. Add unit tests for all telemetry modules
2. Integrate the enhanced telemetry service into the main codebase
3. Update the parsers to use the enhanced telemetry service
4. Add more detailed performance metrics for specific parsing operations
