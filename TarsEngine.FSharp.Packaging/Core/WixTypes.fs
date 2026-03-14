namespace TarsEngine.FSharp.Packaging.Core

open System
open System.Collections.Generic

/// WiX installer configuration
type WixInstallerConfig = {
    ProductName: string
    ProductVersion: string
    ProductCode: Guid
    UpgradeCode: Guid
    Manufacturer: string
    Description: string
    Comments: string
    Keywords: string
    Language: int
    Codepage: int
    SummaryCodepage: int
    Platform: Platform
    InstallerVersion: int
    Compressed: bool
    InstallScope: InstallScope
    InstallPrivileges: InstallPrivileges
}

and Platform =
    | X86
    | X64
    | ARM64

and InstallScope =
    | PerMachine
    | PerUser

and InstallPrivileges =
    | Limited
    | Elevated

/// WiX component configuration
type WixComponent = {
    Id: string
    Guid: Guid
    Directory: string
    Files: WixFile list
    RegistryKeys: WixRegistryKey list
    Shortcuts: WixShortcut list
    Services: WixService list
}

and WixFile = {
    Id: string
    Name: string
    Source: string
    KeyPath: bool
    Vital: bool
    ReadOnly: bool
    Hidden: bool
    System: bool
}

and WixRegistryKey = {
    Root: RegistryRoot
    Key: string
    Name: string option
    Value: string option
    Type: RegistryType
    Action: RegistryAction
}

and RegistryRoot =
    | HKCR  // HKEY_CLASSES_ROOT
    | HKCU  // HKEY_CURRENT_USER
    | HKLM  // HKEY_LOCAL_MACHINE
    | HKU   // HKEY_USERS

and RegistryType =
    | String
    | ExpandableString
    | Integer
    | Binary
    | MultiString

and RegistryAction =
    | Write
    | Append
    | PrependPath
    | AppendPath

and WixShortcut = {
    Id: string
    Name: string
    Description: string
    Target: string
    Arguments: string option
    WorkingDirectory: string option
    Icon: string option
    IconIndex: int
    ShowCommand: ShowCommand
}

and ShowCommand =
    | Normal
    | Minimized
    | Maximized

and WixService = {
    Id: string
    Name: string
    DisplayName: string
    Description: string
    Type: ServiceType
    Start: ServiceStart
    ErrorControl: ServiceErrorControl
    Account: ServiceAccount
    Arguments: string option
    Dependencies: string list
}

and ServiceType =
    | OwnProcess
    | ShareProcess
    | KernelDriver
    | SystemDriver

and ServiceStart =
    | Auto
    | Demand
    | Disabled
    | Boot
    | System

and ServiceErrorControl =
    | Ignore
    | Normal
    | Critical

and ServiceAccount =
    | LocalSystem
    | LocalService
    | NetworkService
    | User of string

/// WiX directory structure
type WixDirectory = {
    Id: string
    Name: string option
    SourceName: string option
    Parent: string option
    Components: WixComponent list
    Subdirectories: WixDirectory list
}

/// WiX feature configuration
type WixFeature = {
    Id: string
    Title: string
    Description: string
    Level: int
    Display: FeatureDisplay
    AllowAdvertise: bool
    InstallDefault: FeatureInstallDefault
    TypicalDefault: FeatureTypicalDefault
    Components: string list
    Features: WixFeature list
}

and FeatureDisplay =
    | Collapse
    | Expand
    | Hidden

and FeatureInstallDefault =
    | Local
    | Source
    | FollowParent

and FeatureTypicalDefault =
    | Advertise
    | Install

/// WiX UI configuration
type WixUI = {
    Id: string
    InstallDirectory: string option
    LicenseFile: string option
    BannerBitmap: string option
    DialogBitmap: string option
    ExclamationIcon: string option
    InfoIcon: string option
    NewIcon: string option
    UpIcon: string option
    CustomActions: WixCustomAction list
}

and WixCustomAction = {
    Id: string
    BinaryKey: string option
    DllEntry: string option
    Execute: CustomActionExecute
    Return: CustomActionReturn
    Impersonate: bool
    Script: CustomActionScript option
}

and CustomActionExecute =
    | Immediate
    | Deferred
    | Rollback
    | Commit

and CustomActionReturn =
    | Check
    | Ignore
    | AsyncWait
    | AsyncNoWait

and CustomActionScript =
    | None
    | VBScript
    | JScript

/// Complete WiX project configuration
type WixProject = {
    Config: WixInstallerConfig
    Directories: WixDirectory list
    Features: WixFeature list
    UI: WixUI option
    Properties: Map<string, string>
    Conditions: WixCondition list
    UpgradeRules: WixUpgrade list
}

and WixCondition = {
    Message: string
    Condition: string
}

and WixUpgrade = {
    Id: Guid
    Minimum: string option
    Maximum: string option
    IncludeMinimum: bool
    IncludeMaximum: bool
    OnlyDetect: bool
    Property: string
}

/// Generated WiX project
type GeneratedWixProject = {
    Project: WixProject
    WxsContent: string
    ProjectFile: string
    BuildScript: string
    OutputDirectory: string
    GeneratedFiles: string list
}

/// WiX project builder for fluent API
type WixProjectBuilder(productName: string) =
    let mutable project = {
        Config = {
            ProductName = productName
            ProductVersion = "1.0.0"
            ProductCode = Guid.NewGuid()
            UpgradeCode = Guid.NewGuid()
            Manufacturer = "TARS"
            Description = $"{productName} Installer"
            Comments = $"Installer for {productName}"
            Keywords = "Installer"
            Language = 1033  // English
            Codepage = 1252
            SummaryCodepage = 1252
            Platform = X64
            InstallerVersion = 500
            Compressed = true
            InstallScope = PerMachine
            InstallPrivileges = Elevated
        }
        Directories = []
        Features = []
        UI = None
        Properties = Map.empty
        Conditions = []
        UpgradeRules = []
    }
    
    member _.Version(version: string) =
        project <- { project with Config = { project.Config with ProductVersion = version } }
        this
    
    member _.Manufacturer(manufacturer: string) =
        project <- { project with Config = { project.Config with Manufacturer = manufacturer } }
        this
    
    member _.Description(description: string) =
        project <- { project with Config = { project.Config with Description = description } }
        this
    
    member _.Platform(platform: Platform) =
        project <- { project with Config = { project.Config with Platform = platform } }
        this
    
    member _.InstallScope(scope: InstallScope) =
        project <- { project with Config = { project.Config with InstallScope = scope } }
        this
    
    member _.Directory(directory: WixDirectory) =
        project <- { project with Directories = directory :: project.Directories }
        this
    
    member _.Feature(feature: WixFeature) =
        project <- { project with Features = feature :: project.Features }
        this
    
    member _.UI(ui: WixUI) =
        project <- { project with UI = Some ui }
        this
    
    member _.Property(key: string, value: string) =
        project <- { project with Properties = project.Properties.Add(key, value) }
        this
    
    member _.Condition(message: string, condition: string) =
        let cond = { Message = message; Condition = condition }
        project <- { project with Conditions = cond :: project.Conditions }
        this
    
    member _.UpgradeRule(upgradeCode: Guid, ?minimum: string, ?maximum: string, ?property: string) =
        let upgrade = {
            Id = upgradeCode
            Minimum = minimum
            Maximum = maximum
            IncludeMinimum = true
            IncludeMaximum = false
            OnlyDetect = false
            Property = defaultArg property "PREVIOUSVERSIONSINSTALLED"
        }
        project <- { project with UpgradeRules = upgrade :: project.UpgradeRules }
        this
    
    member _.Build() = project

/// Helper functions for WiX project creation
module WixHelpers =
    
    /// Creates a new WiX project builder
    let project productName = WixProjectBuilder(productName)
    
    /// Creates a standard directory structure for applications
    let createAppDirectories (appName: string) (files: string list) =
        let programFilesDir = {
            Id = "ProgramFilesFolder"
            Name = None
            SourceName = None
            Parent = None
            Components = []
            Subdirectories = []
        }
        
        let manufacturerDir = {
            Id = "ManufacturerFolder"
            Name = Some "TARS"
            SourceName = None
            Parent = Some "ProgramFilesFolder"
            Components = []
            Subdirectories = []
        }
        
        let appComponents = 
            files
            |> List.mapi (fun i file ->
                {
                    Id = $"Component{i}"
                    Guid = Guid.NewGuid()
                    Directory = "INSTALLFOLDER"
                    Files = [{
                        Id = $"File{i}"
                        Name = System.IO.Path.GetFileName(file)
                        Source = file
                        KeyPath = i = 0
                        Vital = true
                        ReadOnly = false
                        Hidden = false
                        System = false
                    }]
                    RegistryKeys = []
                    Shortcuts = []
                    Services = []
                }
            )
        
        let appDir = {
            Id = "INSTALLFOLDER"
            Name = Some appName
            SourceName = None
            Parent = Some "ManufacturerFolder"
            Components = appComponents
            Subdirectories = []
        }
        
        [programFilesDir; manufacturerDir; appDir]
    
    /// Creates a main feature for the application
    let createMainFeature (appName: string) (componentIds: string list) =
        {
            Id = "MainFeature"
            Title = appName
            Description = $"Main {appName} application"
            Level = 1
            Display = Expand
            AllowAdvertise = false
            InstallDefault = Local
            TypicalDefault = Install
            Components = componentIds
            Features = []
        }
    
    /// Creates standard UI configuration
    let createStandardUI (licenseFile: string option) =
        {
            Id = "WixUI_InstallDir"
            InstallDirectory = Some "INSTALLFOLDER"
            LicenseFile = licenseFile
            BannerBitmap = None
            DialogBitmap = None
            ExclamationIcon = None
            InfoIcon = None
            NewIcon = None
            UpIcon = None
            CustomActions = []
        }
    
    /// Converts platform to string
    let platformToString = function
        | X86 -> "x86"
        | X64 -> "x64"
        | ARM64 -> "arm64"
