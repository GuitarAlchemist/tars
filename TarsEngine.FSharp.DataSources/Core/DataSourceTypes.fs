namespace TarsEngine.FSharp.DataSources.Core

open System

/// Core data source types and interfaces
type DataSourceType =
    | Database of DatabaseType
    | Api of ApiType  
    | File of FileType
    | Stream of StreamType
    | Cache of CacheType
    | Unknown of string

and DatabaseType = PostgreSQL | MySQL | MongoDB | Redis | Elasticsearch
and ApiType = REST | GraphQL | gRPC | WebSocket | SOAP
and FileType = CSV | JSON | XML | Parquet | Binary
and StreamType = Kafka | RabbitMQ | EventHub | RedisStream
and CacheType = Redis | Memcached | InMemory

/// Data source detection result
type DetectionResult = {
    SourceType: DataSourceType
    Confidence: float
    Protocol: string option
    Schema: Map<string, obj> option
    Metadata: Map<string, obj>
}

/// Closure generation parameters
type ClosureParameters = {
    Name: string
    SourceType: DataSourceType
    ConnectionInfo: Map<string, obj>
    Schema: Map<string, obj> option
    Template: string
}

/// Generated closure information
type GeneratedClosure = {
    Name: string
    Code: string
    Parameters: ClosureParameters
    CompiledAssembly: System.Reflection.Assembly option
    ValidationResult: ValidationResult
}

and ValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
}
