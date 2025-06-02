namespace TarsEngine.FSharp.DataSources.Core

open System
open System.Collections.Generic

/// Infrastructure component types supported by TARS
type InfrastructureType =
    | Redis
    | MongoDB
    | MySQL
    | PostgreSQL
    | RabbitMQ
    | Elasticsearch
    | Kafka
    | MinIO
    | Prometheus
    | Grafana

/// Infrastructure component configuration
type InfrastructureConfig = {
    Name: string
    Type: InfrastructureType
    Version: string
    Port: int
    Environment: Map<string, string>
    Volumes: string list
    Networks: string list
    HealthCheck: HealthCheckConfig option
    Dependencies: string list
    CustomConfig: Map<string, obj>
}

and HealthCheckConfig = {
    Command: string
    Interval: TimeSpan
    Timeout: TimeSpan
    Retries: int
    StartPeriod: TimeSpan
}

/// Docker container configuration
type DockerContainerConfig = {
    Image: string
    Tag: string
    ContainerName: string
    Ports: (int * int) list  // (host_port, container_port)
    Environment: Map<string, string>
    Volumes: (string * string) list  // (host_path, container_path)
    Networks: string list
    RestartPolicy: RestartPolicy
    HealthCheck: HealthCheckConfig option
    Command: string option
    WorkingDirectory: string option
    User: string option
    Labels: Map<string, string>
}

and RestartPolicy =
    | No
    | Always
    | OnFailure of int
    | UnlessStopped

/// Infrastructure stack configuration
type InfrastructureStack = {
    Name: string
    Description: string
    Components: InfrastructureConfig list
    Networks: NetworkConfig list
    Volumes: VolumeConfig list
    ComposeVersion: string
    Environment: string  // dev, staging, prod
}

and NetworkConfig = {
    Name: string
    Driver: string
    External: bool
    Labels: Map<string, string>
}

and VolumeConfig = {
    Name: string
    Driver: string
    External: bool
    Labels: Map<string, string>
}

/// Generated infrastructure project
type GeneratedInfrastructure = {
    Stack: InfrastructureStack
    DockerCompose: string
    EnvironmentFiles: Map<string, string>  // filename -> content
    ConfigFiles: Map<string, string>       // filename -> content
    Scripts: Map<string, string>           // filename -> content
    Documentation: string
    OutputDirectory: string
}

/// Infrastructure closure parameters
type InfrastructureClosureParameters = {
    Name: string
    OutputDirectory: string
    Stack: InfrastructureStack
    GenerateScripts: bool
    GenerateDocumentation: bool
    AutoStart: bool
}

/// Infrastructure component builder for fluent API
type InfrastructureBuilder(infraType: InfrastructureType) =
    let mutable config = {
        Name = infraType.ToString().ToLower()
        Type = infraType
        Version = "latest"
        Port = this.GetDefaultPort(infraType)
        Environment = Map.empty
        Volumes = []
        Networks = ["default"]
        HealthCheck = None
        Dependencies = []
        CustomConfig = Map.empty
    }
    
    member _.GetDefaultPort(infraType: InfrastructureType) =
        match infraType with
        | Redis -> 6379
        | MongoDB -> 27017
        | MySQL -> 3306
        | PostgreSQL -> 5432
        | RabbitMQ -> 5672
        | Elasticsearch -> 9200
        | Kafka -> 9092
        | MinIO -> 9000
        | Prometheus -> 9090
        | Grafana -> 3000
    
    member _.Name(name: string) =
        config <- { config with Name = name }
        this
    
    member _.Version(version: string) =
        config <- { config with Version = version }
        this
    
    member _.Port(port: int) =
        config <- { config with Port = port }
        this
    
    member _.Environment(key: string, value: string) =
        config <- { config with Environment = config.Environment.Add(key, value) }
        this
    
    member _.Volume(volume: string) =
        config <- { config with Volumes = volume :: config.Volumes }
        this
    
    member _.Network(network: string) =
        config <- { config with Networks = network :: config.Networks }
        this
    
    member _.HealthCheck(command: string, ?interval: TimeSpan, ?timeout: TimeSpan, ?retries: int) =
        let healthCheck = {
            Command = command
            Interval = defaultArg interval (TimeSpan.FromSeconds(30))
            Timeout = defaultArg timeout (TimeSpan.FromSeconds(10))
            Retries = defaultArg retries 3
            StartPeriod = TimeSpan.FromSeconds(60)
        }
        config <- { config with HealthCheck = Some healthCheck }
        this
    
    member _.DependsOn(dependency: string) =
        config <- { config with Dependencies = dependency :: config.Dependencies }
        this
    
    member _.CustomConfig(key: string, value: obj) =
        config <- { config with CustomConfig = config.CustomConfig.Add(key, value) }
        this
    
    member _.Build() = config

/// Infrastructure stack builder
type InfrastructureStackBuilder(name: string) =
    let mutable stack = {
        Name = name
        Description = $"Infrastructure stack: {name}"
        Components = []
        Networks = []
        Volumes = []
        ComposeVersion = "3.8"
        Environment = "dev"
    }
    
    member _.Description(description: string) =
        stack <- { stack with Description = description }
        this
    
    member _.Component(component: InfrastructureConfig) =
        stack <- { stack with Components = component :: stack.Components }
        this
    
    member _.Network(name: string, ?driver: string, ?external: bool) =
        let network = {
            Name = name
            Driver = defaultArg driver "bridge"
            External = defaultArg external false
            Labels = Map.empty
        }
        stack <- { stack with Networks = network :: stack.Networks }
        this
    
    member _.Volume(name: string, ?driver: string, ?external: bool) =
        let volume = {
            Name = name
            Driver = defaultArg driver "local"
            External = defaultArg external false
            Labels = Map.empty
        }
        stack <- { stack with Volumes = volume :: stack.Volumes }
        this
    
    member _.Environment(env: string) =
        stack <- { stack with Environment = env }
        this
    
    member _.ComposeVersion(version: string) =
        stack <- { stack with ComposeVersion = version }
        this
    
    member _.Build() = stack

/// Helper functions for infrastructure components
module InfrastructureHelpers =
    
    /// Creates a new infrastructure component builder
    let infrastructure infraType = InfrastructureBuilder(infraType)
    
    /// Creates a new infrastructure stack builder
    let stack name = InfrastructureStackBuilder(name)
    
    /// Creates a Redis configuration
    let redis() =
        infrastructure Redis
            .Name("redis")
            .Version("7-alpine")
            .Port(6379)
            .HealthCheck("redis-cli ping")
            .Volume("redis_data:/data")
    
    /// Creates a MongoDB configuration
    let mongodb() =
        infrastructure MongoDB
            .Name("mongodb")
            .Version("6.0")
            .Port(27017)
            .Environment("MONGO_INITDB_ROOT_USERNAME", "admin")
            .Environment("MONGO_INITDB_ROOT_PASSWORD", "password")
            .HealthCheck("echo 'db.runCommand(\"ping\").ok' | mongosh localhost:27017/test --quiet")
            .Volume("mongodb_data:/data/db")
    
    /// Creates a MySQL configuration
    let mysql() =
        infrastructure MySQL
            .Name("mysql")
            .Version("8.0")
            .Port(3306)
            .Environment("MYSQL_ROOT_PASSWORD", "rootpassword")
            .Environment("MYSQL_DATABASE", "appdb")
            .Environment("MYSQL_USER", "appuser")
            .Environment("MYSQL_PASSWORD", "apppassword")
            .HealthCheck("mysqladmin ping -h localhost")
            .Volume("mysql_data:/var/lib/mysql")
    
    /// Creates a PostgreSQL configuration
    let postgresql() =
        infrastructure PostgreSQL
            .Name("postgresql")
            .Version("15-alpine")
            .Port(5432)
            .Environment("POSTGRES_DB", "appdb")
            .Environment("POSTGRES_USER", "appuser")
            .Environment("POSTGRES_PASSWORD", "apppassword")
            .HealthCheck("pg_isready -U appuser -d appdb")
            .Volume("postgresql_data:/var/lib/postgresql/data")
    
    /// Creates a RabbitMQ configuration
    let rabbitmq() =
        infrastructure RabbitMQ
            .Name("rabbitmq")
            .Version("3-management-alpine")
            .Port(5672)
            .Environment("RABBITMQ_DEFAULT_USER", "admin")
            .Environment("RABBITMQ_DEFAULT_PASS", "password")
            .HealthCheck("rabbitmq-diagnostics -q ping")
            .Volume("rabbitmq_data:/var/lib/rabbitmq")
    
    /// Creates an Elasticsearch configuration
    let elasticsearch() =
        infrastructure Elasticsearch
            .Name("elasticsearch")
            .Version("8.8.0")
            .Port(9200)
            .Environment("discovery.type", "single-node")
            .Environment("ES_JAVA_OPTS", "-Xms512m -Xmx512m")
            .HealthCheck("curl -f http://localhost:9200/_cluster/health")
            .Volume("elasticsearch_data:/usr/share/elasticsearch/data")
    
    /// Creates a complete LAMP stack
    let lampStack() =
        stack "lamp"
            .Description("Complete LAMP stack with MySQL, Redis, and monitoring")
            .Component(mysql().Build())
            .Component(redis().Build())
            .Network("lamp_network")
            .Volume("mysql_data")
            .Volume("redis_data")
    
    /// Creates a microservices stack
    let microservicesStack() =
        stack "microservices"
            .Description("Microservices infrastructure with databases, messaging, and monitoring")
            .Component(postgresql().Build())
            .Component(redis().Build())
            .Component(rabbitmq().Build())
            .Component(elasticsearch().Build())
            .Network("microservices_network")
            .Volume("postgresql_data")
            .Volume("redis_data")
            .Volume("rabbitmq_data")
            .Volume("elasticsearch_data")
    
    /// Gets the Docker image name for an infrastructure type
    let getDockerImage infraType version =
        match infraType with
        | Redis -> $"redis:{version}"
        | MongoDB -> $"mongo:{version}"
        | MySQL -> $"mysql:{version}"
        | PostgreSQL -> $"postgres:{version}"
        | RabbitMQ -> $"rabbitmq:{version}"
        | Elasticsearch -> $"docker.elastic.co/elasticsearch/elasticsearch:{version}"
        | Kafka -> $"confluentinc/cp-kafka:{version}"
        | MinIO -> $"minio/minio:{version}"
        | Prometheus -> $"prom/prometheus:{version}"
        | Grafana -> $"grafana/grafana:{version}"
    
    /// Converts infrastructure type to string
    let infraTypeToString = function
        | Redis -> "redis"
        | MongoDB -> "mongodb"
        | MySQL -> "mysql"
        | PostgreSQL -> "postgresql"
        | RabbitMQ -> "rabbitmq"
        | Elasticsearch -> "elasticsearch"
        | Kafka -> "kafka"
        | MinIO -> "minio"
        | Prometheus -> "prometheus"
        | Grafana -> "grafana"
