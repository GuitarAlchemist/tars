namespace TarsEngine.FSharp.DataSources.Closures

open System
open System.IO
open System.Diagnostics
open System.Collections.Generic
open TarsEngine.FSharp.DataSources.Core
open TarsEngine.FSharp.DataSources.Generators

/// Infrastructure closure factory for creating and managing infrastructure components
type InfrastructureClosureFactory() =
    
    let infraGenerator = InfrastructureGenerator()
    let mutable runningStacks = Map.empty<string, Process list>
    
    /// Creates an infrastructure stack closure from metascript configuration
    member _.CreateInfrastructureStackClosure(name: string, config: Map<string, obj>) =
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🏗️ Generating Infrastructure Stack: {name}"
                    
                    // Parse the infrastructure stack configuration
                    let stack = this.ParseInfrastructureStack(name, config)
                    
                    // Generate the infrastructure project
                    let generatedInfra = infraGenerator.GenerateInfrastructureProject(stack, outputDir)
                    
                    printfn $"✅ Infrastructure stack generated successfully at: {outputDir}"
                    printfn $"📊 Components: {stack.Components.Length}"
                    printfn $"🐳 Docker Compose: Generated"
                    printfn $"📜 Scripts: Generated"
                    
                    return Ok {|
                        Type = "INFRASTRUCTURE_STACK"
                        Name = name
                        OutputDirectory = outputDir
                        Components = stack.Components.Length
                        Networks = stack.Networks.Length
                        Volumes = stack.Volumes.Length
                        GeneratedFiles = [
                            "docker-compose.yml"
                            ".env"
                            "start.sh"
                            "stop.sh"
                            "monitor.sh"
                            "README.md"
                        ]
                        Services = stack.Components |> List.map (fun c -> {|
                            Name = c.Name
                            Type = InfrastructureHelpers.infraTypeToString c.Type
                            Port = c.Port
                            Url = $"http://localhost:{c.Port}"
                        |})
                    |}
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate infrastructure stack: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Creates a single component closure
    member _.CreateComponentClosure(componentType: InfrastructureType, name: string, config: Map<string, obj>) =
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🔧 Generating {componentType} Component: {name}"
                    
                    // Create a single-component stack
                    let component = this.ParseInfrastructureComponent(componentType, name, config)
                    let stack = {
                        Name = name
                        Description = $"Single {componentType} component"
                        Components = [component]
                        Networks = []
                        Volumes = []
                        ComposeVersion = "3.8"
                        Environment = "dev"
                    }
                    
                    // Generate the infrastructure project
                    let generatedInfra = infraGenerator.GenerateInfrastructureProject(stack, outputDir)
                    
                    printfn $"✅ {componentType} component generated successfully at: {outputDir}"
                    printfn $"🔗 URL: http://localhost:{component.Port}"
                    
                    return Ok {|
                        Type = "INFRASTRUCTURE_COMPONENT"
                        Name = name
                        ComponentType = InfrastructureHelpers.infraTypeToString componentType
                        OutputDirectory = outputDir
                        Port = component.Port
                        Url = $"http://localhost:{component.Port}"
                        Image = InfrastructureHelpers.getDockerImage componentType component.Version
                    |}
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate {componentType} component: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Creates a predefined stack closure (LAMP, microservices, etc.)
    member _.CreatePredefinedStackClosure(stackType: string, name: string) =
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🏗️ Generating Predefined Stack: {stackType}"
                    
                    let stack = 
                        match stackType.ToUpper() with
                        | "LAMP" -> 
                            InfrastructureHelpers.lampStack()
                                .Name(name)
                                .Description($"LAMP stack: {name}")
                                .Build()
                        | "MICROSERVICES" -> 
                            InfrastructureHelpers.microservicesStack()
                                .Name(name)
                                .Description($"Microservices stack: {name}")
                                .Build()
                        | _ -> failwith $"Unknown predefined stack type: {stackType}"
                    
                    // Generate the infrastructure project
                    let generatedInfra = infraGenerator.GenerateInfrastructureProject(stack, outputDir)
                    
                    printfn $"✅ {stackType} stack generated successfully at: {outputDir}"
                    printfn $"📊 Components: {stack.Components.Length}"
                    
                    return Ok {|
                        Type = "PREDEFINED_STACK"
                        StackType = stackType
                        Name = name
                        OutputDirectory = outputDir
                        Components = stack.Components.Length
                        Services = stack.Components |> List.map (fun c -> {|
                            Name = c.Name
                            Type = InfrastructureHelpers.infraTypeToString c.Type
                            Port = c.Port
                            Url = $"http://localhost:{c.Port}"
                        |})
                    |}
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate {stackType} stack: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Starts an infrastructure stack using Docker Compose
    member _.StartInfrastructureStack(name: string, outputDir: string) =
        async {
            try
                if not (Directory.Exists(outputDir)) then
                    return Error $"Infrastructure directory not found: {outputDir}"
                
                let composeFile = Path.Combine(outputDir, "docker-compose.yml")
                if not (File.Exists(composeFile)) then
                    return Error $"Docker Compose file not found: {composeFile}"
                
                printfn $"🚀 Starting infrastructure stack: {name}"
                
                // Start Docker Compose
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "docker-compose"
                startInfo.Arguments <- "up -d"
                startInfo.WorkingDirectory <- outputDir
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                process.WaitForExit()
                
                if process.ExitCode = 0 then
                    printfn $"✅ Infrastructure stack {name} started successfully"
                    
                    // Get running containers
                    let psInfo = ProcessStartInfo()
                    psInfo.FileName <- "docker-compose"
                    psInfo.Arguments <- "ps"
                    psInfo.WorkingDirectory <- outputDir
                    psInfo.UseShellExecute <- false
                    psInfo.RedirectStandardOutput <- true
                    psInfo.CreateNoWindow <- true
                    
                    let psProcess = Process.Start(psInfo)
                    psProcess.WaitForExit()
                    let output = psProcess.StandardOutput.ReadToEnd()
                    
                    return Ok {|
                        Name = name
                        Status = "running"
                        Output = output
                        StartedAt = DateTime.UtcNow
                    |}
                else
                    let error = process.StandardError.ReadToEnd()
                    return Error $"Failed to start infrastructure stack: {error}"
                    
            with
            | ex -> return Error ex.Message
        }
    
    /// Stops an infrastructure stack
    member _.StopInfrastructureStack(name: string, outputDir: string) =
        async {
            try
                printfn $"🛑 Stopping infrastructure stack: {name}"
                
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "docker-compose"
                startInfo.Arguments <- "down"
                startInfo.WorkingDirectory <- outputDir
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                process.WaitForExit()
                
                if process.ExitCode = 0 then
                    printfn $"✅ Infrastructure stack {name} stopped successfully"
                    return Ok {|
                        Name = name
                        Status = "stopped"
                        StoppedAt = DateTime.UtcNow
                    |}
                else
                    let error = process.StandardError.ReadToEnd()
                    return Error $"Failed to stop infrastructure stack: {error}"
                    
            with
            | ex -> return Error ex.Message
        }
    
    /// Gets available closure types
    member _.GetAvailableClosureTypes() =
        [
            "INFRASTRUCTURE_STACK"
            "INFRASTRUCTURE_COMPONENT"
            "PREDEFINED_STACK"
        ]
    
    /// Gets available component types
    member _.GetAvailableComponentTypes() =
        [
            "REDIS"
            "MONGODB"
            "MYSQL"
            "POSTGRESQL"
            "RABBITMQ"
            "ELASTICSEARCH"
            "KAFKA"
            "MINIO"
            "PROMETHEUS"
            "GRAFANA"
        ]
    
    /// Gets available predefined stacks
    member _.GetAvailablePredefinedStacks() =
        [
            "LAMP"
            "MICROSERVICES"
        ]
    
    /// Creates closure based on type
    member _.CreateClosure(closureType: string, name: string, config: Map<string, obj>) =
        match closureType.ToUpper() with
        | "INFRASTRUCTURE_STACK" -> this.CreateInfrastructureStackClosure(name, config)
        | "INFRASTRUCTURE_COMPONENT" -> 
            let componentType = 
                config.TryFind("component_type") 
                |> Option.map (fun x -> x.ToString().ToUpper())
                |> Option.defaultValue "REDIS"
            let infraType = 
                match componentType with
                | "REDIS" -> Redis
                | "MONGODB" -> MongoDB
                | "MYSQL" -> MySQL
                | "POSTGRESQL" -> PostgreSQL
                | "RABBITMQ" -> RabbitMQ
                | "ELASTICSEARCH" -> Elasticsearch
                | "KAFKA" -> Kafka
                | "MINIO" -> MinIO
                | "PROMETHEUS" -> Prometheus
                | "GRAFANA" -> Grafana
                | _ -> Redis
            this.CreateComponentClosure(infraType, name, config)
        | "PREDEFINED_STACK" ->
            let stackType = 
                config.TryFind("stack_type") 
                |> Option.map (fun x -> x.ToString())
                |> Option.defaultValue "LAMP"
            this.CreatePredefinedStackClosure(stackType, name)
        | _ -> failwith $"Unknown closure type: {closureType}"
    
    /// Parses infrastructure stack from configuration
    member _.ParseInfrastructureStack(name: string, config: Map<string, obj>) =
        // TODO: Implement proper parsing from metascript configuration
        // For now, return a sample microservices stack
        InfrastructureHelpers.microservicesStack()
            .Name(name)
            .Description($"Infrastructure stack: {name}")
            .Build()
    
    /// Parses infrastructure component from configuration
    member _.ParseInfrastructureComponent(componentType: InfrastructureType, name: string, config: Map<string, obj>) =
        let builder = InfrastructureHelpers.infrastructure componentType
        
        let mutable component = builder.Name(name)
        
        // Parse version
        if config.ContainsKey("version") then
            component <- component.Version(config.["version"].ToString())
        
        // Parse port
        if config.ContainsKey("port") then
            component <- component.Port(int (config.["port"].ToString()))
        
        // Parse environment variables
        if config.ContainsKey("environment") then
            // TODO: Parse environment variables from config
            ()
        
        component.Build()
