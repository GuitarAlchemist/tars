namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services

/// <summary>
/// Configuration Command - Manages TARS configuration and settings
/// Provides real configuration management functionality
/// </summary>
type ConfigCommand(logger: ILogger<ConfigCommand>) =
    
    interface ICommand with
        member _.Name = "config"
        member _.Description = "Manage TARS configuration and settings"
        member self.Usage = "tars config <subcommand> [options]"
        member self.Examples = [
            "tars config show"
            "tars config set log_level debug"
            "tars config get log_level"
            "tars config init"
        ]
        member self.ValidateOptions(_) = true

        member self.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    match options.Arguments with
                    | [] ->
                        self.ShowConfigHelp()
                        return CommandResult.success "Help displayed"
                    | "show" :: _ ->
                        let result = self.ShowCurrentConfig()
                        return if result = 0 then CommandResult.success "Config shown" else CommandResult.failure "Failed to show config"
                    | "set" :: key :: value :: _ ->
                        let result = self.SetConfigValue(key, value)
                        return if result = 0 then CommandResult.success "Config set" else CommandResult.failure "Failed to set config"
                    | "get" :: key :: _ ->
                        let result = self.GetConfigValue(key)
                        return if result = 0 then CommandResult.success "Config retrieved" else CommandResult.failure "Failed to get config"
                    | "list" :: _ ->
                        let result = self.ListAllConfig()
                        return if result = 0 then CommandResult.success "Config listed" else CommandResult.failure "Failed to list config"
                    | "reset" :: _ ->
                        let result = self.ResetConfig()
                        return if result = 0 then CommandResult.success "Config reset" else CommandResult.failure "Failed to reset config"
                    | "init" :: _ ->
                        let result = self.InitializeConfig()
                        return if result = 0 then CommandResult.success "Config initialized" else CommandResult.failure "Failed to initialize config"
                    | unknown :: _ ->
                        logger.LogWarning("Invalid config command: {Command}", String.Join(" ", unknown))
                        self.ShowConfigHelp()
                        return CommandResult.failure $"Unknown subcommand: {unknown}"
                with
                | ex ->
                    logger.LogError(ex, "Error executing config command")
                    printfn $"❌ Config command failed: {ex.Message}"
                    return CommandResult.failure ex.Message
            }
    
    /// <summary>
    /// Shows configuration command help
    /// </summary>
    member self.ShowConfigHelp() =
        printfn "TARS Configuration Management"
        printfn "============================"
        printfn ""
        printfn "Available Commands:"
        printfn "  show              - Show current configuration"
        printfn "  set <key> <value> - Set configuration value"
        printfn "  get <key>         - Get configuration value"
        printfn "  list              - List all configuration keys"
        printfn "  reset             - Reset to default configuration"
        printfn "  init              - Initialize default configuration"
        printfn ""
        printfn "Usage: tars config [command]"
        printfn ""
        printfn "Examples:"
        printfn "  tars config set log_level debug"
        printfn "  tars config get log_level"
        printfn "  tars config show"
        printfn ""
        printfn "Configuration is stored in: .tars/config/settings.json"
    
    /// <summary>
    /// Gets the configuration file path
    /// </summary>
    member self.GetConfigPath() =
        let configDir = ".tars/config"
        Directory.CreateDirectory(configDir) |> ignore
        Path.Combine(configDir, "settings.json")
    
    /// <summary>
    /// Loads configuration from file
    /// </summary>
    member self.LoadConfig() =
        try
            let configFile = self.GetConfigPath()
            if File.Exists(configFile) then
                let configJson = File.ReadAllText(configFile)
                JsonSerializer.Deserialize<Map<string, string>>(configJson)
            else
                Map.empty<string, string>
        with
        | ex ->
            logger.LogError(ex, "Error loading configuration")
            Map.empty<string, string>

    /// <summary>
    /// Saves configuration to file
    /// </summary>
    member self.SaveConfig(config: Map<string, string>) =
        try
            let configFile = self.GetConfigPath()
            let configJson = JsonSerializer.Serialize(config, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(configFile, configJson)
            true
        with
        | ex ->
            logger.LogError(ex, "Error saving configuration")
            printfn $"❌ Failed to save configuration: {ex.Message}"
            false

    /// <summary>
    /// Shows current configuration
    /// </summary>
    member self.ShowCurrentConfig() =
        printfn "CURRENT TARS CONFIGURATION"
        printfn "========================="

        try
            let config = self.LoadConfig()

            if config.IsEmpty then
                printfn "No configuration found."
                printfn "Run 'tars config init' to create default configuration."
            else
                printfn "Configuration Settings:"
                for kvp in config do
                    printfn $"  {kvp.Key} = {kvp.Value}"

                printfn ""
                printfn $"Configuration file: {self.GetConfigPath()}"
            
            0
            
        with
        | ex ->
            logger.LogError(ex, "Error showing configuration")
            printfn $"❌ Failed to show configuration: {ex.Message}"
            1
    
    /// <summary>
    /// Sets a configuration value
    /// </summary>
    member self.SetConfigValue(key: string, value: string) =
        printfn $"Setting configuration: {key} = {value}"

        try
            let config = self.LoadConfig()
            let updatedConfig = config.Add(key, value)

            if self.SaveConfig(updatedConfig) then
                printfn $"✅ Configuration updated: {key} = {value}"
                0
            else
                1

        with
        | ex ->
            logger.LogError(ex, "Error setting configuration value")
            printfn $"❌ Failed to set configuration: {ex.Message}"
            1

    /// <summary>
    /// Gets a configuration value
    /// </summary>
    member self.GetConfigValue(key: string) =
        try
            let config = self.LoadConfig()

            match config.TryFind(key) with
            | Some value ->
                printfn $"{key} = {value}"
                0
            | None ->
                printfn $"Configuration key '{key}' not found."
                printfn "Available keys:"
                for k in config.Keys do
                    printfn $"  {k}"
                1

        with
        | ex ->
            logger.LogError(ex, "Error getting configuration value")
            printfn $"❌ Failed to get configuration: {ex.Message}"
            1

    /// <summary>
    /// Lists all configuration keys
    /// </summary>
    member self.ListAllConfig() =
        printfn "ALL CONFIGURATION KEYS"
        printfn "======================"

        try
            let config = self.LoadConfig()

            if config.IsEmpty then
                printfn "No configuration keys found."
                printfn "Run 'tars config init' to create default configuration."
            else
                printfn "Available configuration keys:"
                for key in config.Keys do
                    printfn $"  {key}"

                printfn ""
                printfn $"Total keys: {config.Count}"

            0

        with
        | ex ->
            logger.LogError(ex, "Error listing configuration")
            printfn $"❌ Failed to list configuration: {ex.Message}"
            1
    
    /// <summary>
    /// Resets configuration to defaults
    /// </summary>
    member self.ResetConfig() =
        printfn "RESETTING CONFIGURATION TO DEFAULTS"
        printfn "==================================="

        try
            let defaultConfig = Map.ofList [
                ("log_level", "info")
                ("output_format", "console")
                ("max_parallel_tasks", "4")
                ("timeout_seconds", "30")
                ("auto_save", "true")
            ]

            if self.SaveConfig(defaultConfig) then
                printfn "✅ Configuration reset to defaults:"
                for kvp in defaultConfig do
                    printfn $"  {kvp.Key} = {kvp.Value}"
                0
            else
                1

        with
        | ex ->
            logger.LogError(ex, "Error resetting configuration")
            printfn $"❌ Failed to reset configuration: {ex.Message}"
            1

    /// <summary>
    /// Initializes default configuration
    /// </summary>
    member self.InitializeConfig() =
        printfn "INITIALIZING DEFAULT CONFIGURATION"
        printfn "================================="

        try
            let configFile = self.GetConfigPath()

            if File.Exists(configFile) then
                printfn "Configuration already exists."
                printfn "Use 'tars config reset' to reset to defaults."
                printfn "Use 'tars config show' to view current configuration."
                0
            else
                self.ResetConfig()

        with
        | ex ->
            logger.LogError(ex, "Error initializing configuration")
            printfn $"❌ Failed to initialize configuration: {ex.Message}"
            1
