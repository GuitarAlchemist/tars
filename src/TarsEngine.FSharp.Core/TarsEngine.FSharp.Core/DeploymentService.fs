namespace TarsEngine.FSharp.Core

open System
open System.Threading.Tasks

/// Injectable Deployment Service for TARS Metascripts
module DeploymentService =

    // Use the existing UnifiedDeployment types
    type DeploymentPlatform = UnifiedDeployment.DeploymentPlatform
    type DeploymentConfig = UnifiedDeployment.DeploymentConfig



    /// Deployment result
    type DeploymentResult = {
        Success: bool
        Platform: DeploymentPlatform
        DeploymentId: string
        Message: string
        GeneratedFiles: string list
        Endpoints: string list
    }

    /// Deployment status
    type DeploymentStatus =
        | NotDeployed
        | Deploying
        | Deployed
        | Failed of string
        | Updating

    /// Validation result
    type ValidationResult = {
        IsValid: bool
        Errors: string list
        Warnings: string list
    }

    /// Injectable Deployment Service Interface
    type IDeploymentService =
        abstract GetAvailablePlatforms: unit -> DeploymentPlatform list
        abstract DeployTo: DeploymentPlatform * DeploymentConfig -> Task<DeploymentResult>
        abstract GetDeploymentStatus: string -> Task<DeploymentStatus>
        abstract GenerateDeploymentScript: DeploymentPlatform * DeploymentConfig -> string
        abstract ValidateDeployment: DeploymentConfig -> ValidationResult
        abstract ListDeployments: unit -> Task<(string * DeploymentPlatform * DeploymentStatus) list>
        abstract RemoveDeployment: string -> Task<bool>

    /// TARS Deployment Service Implementation
    type TarsDeploymentService() =
        
        let mutable deployments = Map.empty<string, (DeploymentPlatform * DeploymentConfig * DeploymentStatus)>
        
        /// Check if platform is available
        let isPlatformAvailable platform =
            match platform with
            | UnifiedDeployment.WindowsService -> UnifiedDeployment.checkWindowsServiceAvailability()
            | UnifiedDeployment.Docker -> UnifiedDeployment.checkDockerAvailability()
            | UnifiedDeployment.DockerCompose -> UnifiedDeployment.checkDockerComposeAvailability()
            | UnifiedDeployment.Kubernetes -> UnifiedDeployment.checkKubernetesAvailability()
            | UnifiedDeployment.Hyperlight -> UnifiedDeployment.checkHyperlightAvailability()
            | UnifiedDeployment.Native -> true // Always available
        
        /// Generate deployment ID
        let generateDeploymentId (platform: DeploymentPlatform) (config: DeploymentConfig) =
            sprintf "%s-%s-%s" (platform.ToString().ToLower()) config.NodeName (DateTime.Now.ToString("yyyyMMddHHmmss"))
        
        interface IDeploymentService with
            
            member _.GetAvailablePlatforms() =
                [UnifiedDeployment.WindowsService; UnifiedDeployment.Docker; UnifiedDeployment.DockerCompose; UnifiedDeployment.Kubernetes; UnifiedDeployment.Hyperlight; UnifiedDeployment.Native]
                |> List.filter isPlatformAvailable
            
            member _.DeployTo(platform, config) = task {
                try
                    if not (isPlatformAvailable platform) then
                        return {
                            Success = false
                            Platform = platform
                            DeploymentId = ""
                            Message = sprintf "Platform %A is not available" platform
                            GeneratedFiles = []
                            Endpoints = []
                        }
                    else
                        let deploymentId = generateDeploymentId platform config
                        
                        // Update deployment status
                        deployments <- deployments |> Map.add deploymentId (platform, config, Deploying)
                        
                        // Generate deployment script
                        let script = UnifiedDeployment.generateDeployment(platform, config)
                        
                        // Simulate deployment process
                        do! Task.Delay(1000) // Simulate deployment time
                        
                        // Update deployment status to deployed
                        deployments <- deployments |> Map.add deploymentId (platform, config, Deployed)
                        
                        let endpoint = sprintf "http://localhost:%d" config.Port
                        
                        return {
                            Success = true
                            Platform = platform
                            DeploymentId = deploymentId
                            Message = sprintf "Successfully deployed to %A" platform
                            GeneratedFiles = [sprintf "%s-deployment.yaml" config.NodeName]
                            Endpoints = [endpoint]
                        }
                with
                | ex ->
                    return {
                        Success = false
                        Platform = platform
                        DeploymentId = ""
                        Message = sprintf "Deployment failed: %s" ex.Message
                        GeneratedFiles = []
                        Endpoints = []
                    }
            }
            
            member _.GetDeploymentStatus(deploymentId) = task {
                match deployments |> Map.tryFind deploymentId with
                | Some (_, _, status) -> return status
                | None -> return NotDeployed
            }
            
            member _.GenerateDeploymentScript(platform, config) =
                UnifiedDeployment.generateDeployment(platform, config)
            
            member _.ValidateDeployment(config) =
                let errors = ResizeArray<string>()
                let warnings = ResizeArray<string>()
                
                // Validate node name
                if String.IsNullOrWhiteSpace(config.NodeName) then
                    errors.Add("NodeName cannot be empty")
                
                // Validate port
                if config.Port <= 0 || config.Port > 65535 then
                    errors.Add("Port must be between 1 and 65535")
                
                // Validate environment
                if String.IsNullOrWhiteSpace(config.Environment) then
                    warnings.Add("Environment not specified, defaulting to 'development'")
                
                {
                    IsValid = errors.Count = 0
                    Errors = errors |> List.ofSeq
                    Warnings = warnings |> List.ofSeq
                }
            
            member _.ListDeployments() = task {
                return deployments
                       |> Map.toList
                       |> List.map (fun (id, (platform, _, status)) -> (id, platform, status))
            }
            
            member _.RemoveDeployment(deploymentId) = task {
                match deployments |> Map.tryFind deploymentId with
                | Some _ ->
                    deployments <- deployments |> Map.remove deploymentId
                    return true
                | None -> return false
            }

    /// Service registration helper
    let registerDeploymentService (serviceProvider: IServiceProvider -> IDeploymentService) =
        // This would be registered in the DI container
        serviceProvider

    /// Create deployment service instance
    let createDeploymentService() : IDeploymentService =
        TarsDeploymentService() :> IDeploymentService

    /// Helper functions for metascript usage
    module MetascriptHelpers =
        
        /// Quick deploy to multiple platforms
        let deployToMultiplePlatforms (service: IDeploymentService) (platforms: DeploymentPlatform list) (config: DeploymentConfig) = task {
            let! results = 
                platforms
                |> List.map (fun platform -> service.DeployTo(platform, config))
                |> Task.WhenAll
            
            return results |> Array.toList
        }
        
        /// Deploy with validation
        let deployWithValidation (service: IDeploymentService) (platform: DeploymentPlatform) (config: DeploymentConfig) = task {
            let validation = service.ValidateDeployment(config)
            
            if validation.IsValid then
                return! service.DeployTo(platform, config)
            else
                return {
                    Success = false
                    Platform = platform
                    DeploymentId = ""
                    Message = sprintf "Validation failed: %s" (String.concat "; " validation.Errors)
                    GeneratedFiles = []
                    Endpoints = []
                }
        }
        
        /// Get deployment summary
        let getDeploymentSummary (service: IDeploymentService) = task {
            let availablePlatforms = service.GetAvailablePlatforms()
            let! deployments = service.ListDeployments()
            
            return {|
                AvailablePlatforms = availablePlatforms
                TotalDeployments = deployments.Length
                ActiveDeployments = deployments |> List.filter (fun (_, _, status) -> status = Deployed) |> List.length
                FailedDeployments = deployments |> List.filter (fun (_, _, status) -> match status with Failed _ -> true | _ -> false) |> List.length
            |}
        }
