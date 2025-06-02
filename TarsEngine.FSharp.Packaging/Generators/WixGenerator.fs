namespace TarsEngine.FSharp.Packaging.Generators

open System
open System.IO
open System.Text
open TarsEngine.FSharp.Packaging.Core

/// WiX installer generator for creating MSI packages
type WixGenerator() =
    
    /// Generates WiX source file (.wxs)
    member _.GenerateWxsFile(project: WixProject) =
        let sb = StringBuilder()
        
        // XML header
        sb.AppendLine("<?xml version=\"1.0\" encoding=\"UTF-8\"?>") |> ignore
        sb.AppendLine("<Wix xmlns=\"http://schemas.microsoft.com/wix/2006/wi\">") |> ignore
        
        // Product element
        this.GenerateProductElement(sb, project)
        
        // Package element
        this.GeneratePackageElement(sb, project.Config)
        
        // Media element
        sb.AppendLine("    <Media Id=\"1\" Cabinet=\"media1.cab\" EmbedCab=\"yes\" />") |> ignore
        sb.AppendLine() |> ignore
        
        // Properties
        this.GenerateProperties(sb, project.Properties)
        
        // Conditions
        this.GenerateConditions(sb, project.Conditions)
        
        // Upgrade rules
        this.GenerateUpgradeRules(sb, project.UpgradeRules)
        
        // Directory structure
        this.GenerateDirectoryStructure(sb, project.Directories)
        
        // Features
        this.GenerateFeatures(sb, project.Features)
        
        // UI
        match project.UI with
        | Some ui -> this.GenerateUI(sb, ui)
        | None -> ()
        
        // Install execute sequence
        this.GenerateInstallExecuteSequence(sb)
        
        // Close Product and Wix elements
        sb.AppendLine("  </Product>") |> ignore
        sb.AppendLine("</Wix>") |> ignore
        
        sb.ToString()
    
    /// Generates Product element
    member _.GenerateProductElement(sb: StringBuilder, project: WixProject) =
        let config = project.Config
        
        sb.AppendLine($"  <Product Id=\"{config.ProductCode}\"") |> ignore
        sb.AppendLine($"           Name=\"{config.ProductName}\"") |> ignore
        sb.AppendLine($"           Language=\"{config.Language}\"") |> ignore
        sb.AppendLine($"           Version=\"{config.ProductVersion}\"") |> ignore
        sb.AppendLine($"           Manufacturer=\"{config.Manufacturer}\"") |> ignore
        sb.AppendLine($"           UpgradeCode=\"{config.UpgradeCode}\">") |> ignore
        sb.AppendLine() |> ignore
    
    /// Generates Package element
    member _.GeneratePackageElement(sb: StringBuilder, config: WixInstallerConfig) =
        sb.AppendLine("    <Package InstallerVersion=\"500\"") |> ignore
        sb.AppendLine("             Compressed=\"yes\"") |> ignore
        sb.AppendLine($"             InstallScope=\"{this.InstallScopeToString(config.InstallScope)}\"") |> ignore
        sb.AppendLine($"             Platform=\"{WixHelpers.platformToString config.Platform}\"") |> ignore
        sb.AppendLine($"             Description=\"{config.Description}\"") |> ignore
        sb.AppendLine($"             Comments=\"{config.Comments}\"") |> ignore
        sb.AppendLine($"             Keywords=\"{config.Keywords}\" />") |> ignore
        sb.AppendLine() |> ignore
    
    /// Generates Properties section
    member _.GenerateProperties(sb: StringBuilder, properties: Map<string, string>) =
        if not properties.IsEmpty then
            for kvp in properties do
                sb.AppendLine($"    <Property Id=\"{kvp.Key}\" Value=\"{kvp.Value}\" />") |> ignore
            sb.AppendLine() |> ignore
    
    /// Generates Conditions section
    member _.GenerateConditions(sb: StringBuilder, conditions: WixCondition list) =
        for condition in conditions do
            sb.AppendLine($"    <Condition Message=\"{condition.Message}\">") |> ignore
            sb.AppendLine($"      {condition.Condition}") |> ignore
            sb.AppendLine("    </Condition>") |> ignore
        
        if not conditions.IsEmpty then
            sb.AppendLine() |> ignore
    
    /// Generates Upgrade rules
    member _.GenerateUpgradeRules(sb: StringBuilder, upgrades: WixUpgrade list) =
        for upgrade in upgrades do
            sb.AppendLine($"    <Upgrade Id=\"{upgrade.Id}\">") |> ignore
            sb.AppendLine($"      <UpgradeVersion Minimum=\"{upgrade.Minimum |> Option.defaultValue "0.0.0"}\"") |> ignore
            sb.AppendLine($"                      Maximum=\"{upgrade.Maximum |> Option.defaultValue "999.999.999"}\"") |> ignore
            sb.AppendLine($"                      Property=\"{upgrade.Property}\"") |> ignore
            sb.AppendLine($"                      IncludeMinimum=\"{if upgrade.IncludeMinimum then "yes" else "no"}\"") |> ignore
            sb.AppendLine($"                      IncludeMaximum=\"{if upgrade.IncludeMaximum then "yes" else "no"}\" />") |> ignore
            sb.AppendLine("    </Upgrade>") |> ignore
        
        if not upgrades.IsEmpty then
            sb.AppendLine() |> ignore
    
    /// Generates Directory structure
    member _.GenerateDirectoryStructure(sb: StringBuilder, directories: WixDirectory list) =
        sb.AppendLine("    <Directory Id=\"TARGETDIR\" Name=\"SourceDir\">") |> ignore
        
        for directory in directories do
            this.GenerateDirectory(sb, directory, 2)
        
        sb.AppendLine("    </Directory>") |> ignore
        sb.AppendLine() |> ignore
    
    /// Generates a single directory
    member _.GenerateDirectory(sb: StringBuilder, directory: WixDirectory, indent: int) =
        let indentStr = String(' ', indent * 2)
        
        sb.AppendLine($"{indentStr}<Directory Id=\"{directory.Id}\"") |> ignore
        
        match directory.Name with
        | Some name -> sb.AppendLine($"{indentStr}           Name=\"{name}\">") |> ignore
        | None -> sb.AppendLine($"{indentStr}>") |> ignore
        
        // Generate components
        for component in directory.Components do
            this.GenerateComponent(sb, component, indent + 1)
        
        // Generate subdirectories
        for subdir in directory.Subdirectories do
            this.GenerateDirectory(sb, subdir, indent + 1)
        
        sb.AppendLine($"{indentStr}</Directory>") |> ignore
    
    /// Generates a component
    member _.GenerateComponent(sb: StringBuilder, component: WixComponent, indent: int) =
        let indentStr = String(' ', indent * 2)
        
        sb.AppendLine($"{indentStr}<Component Id=\"{component.Id}\" Guid=\"{component.Guid}\">") |> ignore
        
        // Generate files
        for file in component.Files do
            sb.AppendLine($"{indentStr}  <File Id=\"{file.Id}\"") |> ignore
            sb.AppendLine($"{indentStr}        Name=\"{file.Name}\"") |> ignore
            sb.AppendLine($"{indentStr}        Source=\"{file.Source}\"") |> ignore
            if file.KeyPath then
                sb.AppendLine($"{indentStr}        KeyPath=\"yes\"") |> ignore
            sb.AppendLine($"{indentStr}        Vital=\"{if file.Vital then "yes" else "no"}\" />") |> ignore
        
        // Generate registry keys
        for regKey in component.RegistryKeys do
            this.GenerateRegistryKey(sb, regKey, indent + 1)
        
        // Generate shortcuts
        for shortcut in component.Shortcuts do
            this.GenerateShortcut(sb, shortcut, indent + 1)
        
        sb.AppendLine($"{indentStr}</Component>") |> ignore
    
    /// Generates registry key
    member _.GenerateRegistryKey(sb: StringBuilder, regKey: WixRegistryKey, indent: int) =
        let indentStr = String(' ', indent * 2)
        
        sb.AppendLine($"{indentStr}<RegistryKey Root=\"{this.RegistryRootToString(regKey.Root)}\"") |> ignore
        sb.AppendLine($"{indentStr}             Key=\"{regKey.Key}\">") |> ignore
        
        match regKey.Name, regKey.Value with
        | Some name, Some value ->
            sb.AppendLine($"{indentStr}  <RegistryValue Name=\"{name}\"") |> ignore
            sb.AppendLine($"{indentStr}                 Type=\"{this.RegistryTypeToString(regKey.Type)}\"") |> ignore
            sb.AppendLine($"{indentStr}                 Value=\"{value}\" />") |> ignore
        | _ -> ()
        
        sb.AppendLine($"{indentStr}</RegistryKey>") |> ignore
    
    /// Generates shortcut
    member _.GenerateShortcut(sb: StringBuilder, shortcut: WixShortcut, indent: int) =
        let indentStr = String(' ', indent * 2)
        
        sb.AppendLine($"{indentStr}<Shortcut Id=\"{shortcut.Id}\"") |> ignore
        sb.AppendLine($"{indentStr}          Name=\"{shortcut.Name}\"") |> ignore
        sb.AppendLine($"{indentStr}          Description=\"{shortcut.Description}\"") |> ignore
        sb.AppendLine($"{indentStr}          Target=\"{shortcut.Target}\"") |> ignore
        
        match shortcut.Arguments with
        | Some args -> sb.AppendLine($"{indentStr}          Arguments=\"{args}\"") |> ignore
        | None -> ()
        
        match shortcut.WorkingDirectory with
        | Some workDir -> sb.AppendLine($"{indentStr}          WorkingDirectory=\"{workDir}\"") |> ignore
        | None -> ()
        
        sb.AppendLine($"{indentStr}          ShowCmd=\"{this.ShowCommandToString(shortcut.ShowCommand)}\" />") |> ignore
    
    /// Generates Features section
    member _.GenerateFeatures(sb: StringBuilder, features: WixFeature list) =
        for feature in features do
            this.GenerateFeature(sb, feature, 2)
        
        if not features.IsEmpty then
            sb.AppendLine() |> ignore
    
    /// Generates a single feature
    member _.GenerateFeature(sb: StringBuilder, feature: WixFeature, indent: int) =
        let indentStr = String(' ', indent * 2)
        
        sb.AppendLine($"{indentStr}<Feature Id=\"{feature.Id}\"") |> ignore
        sb.AppendLine($"{indentStr}         Title=\"{feature.Title}\"") |> ignore
        sb.AppendLine($"{indentStr}         Description=\"{feature.Description}\"") |> ignore
        sb.AppendLine($"{indentStr}         Level=\"{feature.Level}\">") |> ignore
        
        // Component references
        for componentId in feature.Components do
            sb.AppendLine($"{indentStr}  <ComponentRef Id=\"{componentId}\" />") |> ignore
        
        // Sub-features
        for subFeature in feature.Features do
            this.GenerateFeature(sb, subFeature, indent + 1)
        
        sb.AppendLine($"{indentStr}</Feature>") |> ignore
    
    /// Generates UI section
    member _.GenerateUI(sb: StringBuilder, ui: WixUI) =
        sb.AppendLine($"    <UIRef Id=\"{ui.Id}\" />") |> ignore
        
        match ui.InstallDirectory with
        | Some dir ->
            sb.AppendLine($"    <Property Id=\"WIXUI_INSTALLDIR\" Value=\"{dir}\" />") |> ignore
        | None -> ()
        
        match ui.LicenseFile with
        | Some license ->
            sb.AppendLine($"    <WixVariable Id=\"WixUILicenseRtf\" Value=\"{license}\" />") |> ignore
        | None -> ()
        
        sb.AppendLine() |> ignore
    
    /// Generates InstallExecuteSequence
    member _.GenerateInstallExecuteSequence(sb: StringBuilder) =
        sb.AppendLine("    <InstallExecuteSequence>") |> ignore
        sb.AppendLine("      <RemoveExistingProducts After=\"InstallValidate\" />") |> ignore
        sb.AppendLine("    </InstallExecuteSequence>") |> ignore
        sb.AppendLine() |> ignore
    
    /// Generates WiX project file (.wixproj)
    member _.GenerateWixProjectFile(project: WixProject, wxsFileName: string) =
        let sb = StringBuilder()
        
        sb.AppendLine("<?xml version=\"1.0\" encoding=\"utf-8\"?>") |> ignore
        sb.AppendLine("<Project ToolsVersion=\"4.0\" DefaultTargets=\"Build\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">") |> ignore
        sb.AppendLine("  <PropertyGroup>") |> ignore
        sb.AppendLine("    <Configuration Condition=\" '$(Configuration)' == '' \">Debug</Configuration>") |> ignore
        sb.AppendLine("    <Platform Condition=\" '$(Platform)' == '' \">x86</Platform>") |> ignore
        sb.AppendLine("    <ProductVersion>3.10</ProductVersion>") |> ignore
        sb.AppendLine("    <ProjectGuid>{" + Guid.NewGuid().ToString().ToUpper() + "}</ProjectGuid>") |> ignore
        sb.AppendLine("    <SchemaVersion>2.0</SchemaVersion>") |> ignore
        sb.AppendLine($"    <OutputName>{project.Config.ProductName}</OutputName>") |> ignore
        sb.AppendLine("    <OutputType>Package</OutputType>") |> ignore
        sb.AppendLine("    <WixTargetsPath Condition=\" '$(WixTargetsPath)' == '' AND '$(MSBuildExtensionsPath32)' != '' \">$(MSBuildExtensionsPath32)\\Microsoft\\WiX\\v3.x\\Wix.targets</WixTargetsPath>") |> ignore
        sb.AppendLine("    <WixTargetsPath Condition=\" '$(WixTargetsPath)' == '' \">$(MSBuildExtensionsPath)\\Microsoft\\WiX\\v3.x\\Wix.targets</WixTargetsPath>") |> ignore
        sb.AppendLine("  </PropertyGroup>") |> ignore
        sb.AppendLine("  <PropertyGroup Condition=\" '$(Configuration)|$(Platform)' == 'Debug|x86' \">") |> ignore
        sb.AppendLine("    <OutputPath>bin\\$(Configuration)\\</OutputPath>") |> ignore
        sb.AppendLine("    <IntermediateOutputPath>obj\\$(Configuration)\\</IntermediateOutputPath>") |> ignore
        sb.AppendLine("    <DefineConstants>Debug</DefineConstants>") |> ignore
        sb.AppendLine("  </PropertyGroup>") |> ignore
        sb.AppendLine("  <PropertyGroup Condition=\" '$(Configuration)|$(Platform)' == 'Release|x86' \">") |> ignore
        sb.AppendLine("    <OutputPath>bin\\$(Configuration)\\</OutputPath>") |> ignore
        sb.AppendLine("    <IntermediateOutputPath>obj\\$(Configuration)\\</IntermediateOutputPath>") |> ignore
        sb.AppendLine("  </PropertyGroup>") |> ignore
        sb.AppendLine("  <ItemGroup>") |> ignore
        sb.AppendLine($"    <Compile Include=\"{wxsFileName}\" />") |> ignore
        sb.AppendLine("  </ItemGroup>") |> ignore
        sb.AppendLine("  <Import Project=\"$(WixTargetsPath)\" />") |> ignore
        sb.AppendLine("</Project>") |> ignore
        
        sb.ToString()
    
    /// Generates build script
    member _.GenerateBuildScript(project: WixProject, projectFileName: string) =
        let sb = StringBuilder()
        
        sb.AppendLine("@echo off") |> ignore
        sb.AppendLine($"REM Build script for {project.Config.ProductName} installer") |> ignore
        sb.AppendLine("REM Generated by TARS WiX Generator") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("echo Building WiX installer...") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("REM Check if WiX is installed") |> ignore
        sb.AppendLine("where candle >nul 2>nul") |> ignore
        sb.AppendLine("if %ERRORLEVEL% NEQ 0 (") |> ignore
        sb.AppendLine("    echo ERROR: WiX Toolset not found. Please install WiX Toolset.") |> ignore
        sb.AppendLine("    echo Download from: https://wixtoolset.org/releases/") |> ignore
        sb.AppendLine("    exit /b 1") |> ignore
        sb.AppendLine(")") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("REM Build using MSBuild") |> ignore
        sb.AppendLine($"msbuild {projectFileName} /p:Configuration=Release /p:Platform=x86") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("if %ERRORLEVEL% EQU 0 (") |> ignore
        sb.AppendLine($"    echo SUCCESS: {project.Config.ProductName} installer built successfully!") |> ignore
        sb.AppendLine("    echo Output: bin\\Release\\") |> ignore
        sb.AppendLine(") else (") |> ignore
        sb.AppendLine("    echo ERROR: Build failed!") |> ignore
        sb.AppendLine("    exit /b 1") |> ignore
        sb.AppendLine(")") |> ignore
        
        sb.ToString()
    
    /// Helper methods for converting enums to strings
    member _.InstallScopeToString = function
        | PerMachine -> "perMachine"
        | PerUser -> "perUser"
    
    member _.RegistryRootToString = function
        | HKCR -> "HKCR"
        | HKCU -> "HKCU"
        | HKLM -> "HKLM"
        | HKU -> "HKU"
    
    member _.RegistryTypeToString = function
        | String -> "string"
        | ExpandableString -> "expandable"
        | Integer -> "integer"
        | Binary -> "binary"
        | MultiString -> "multiString"
    
    member _.ShowCommandToString = function
        | Normal -> "normal"
        | Minimized -> "minimized"
        | Maximized -> "maximized"

    /// Generates complete WiX project
    member _.GenerateWixProject(project: WixProject, outputDir: string) =
        // Create output directory
        Directory.CreateDirectory(outputDir) |> ignore

        // Generate files
        let wxsContent = this.GenerateWxsFile(project)
        let wxsFileName = $"{project.Config.ProductName}.wxs"
        let projectContent = this.GenerateWixProjectFile(project, wxsFileName)
        let projectFileName = $"{project.Config.ProductName}.wixproj"
        let buildScript = this.GenerateBuildScript(project, projectFileName)

        // Write files
        File.WriteAllText(Path.Combine(outputDir, wxsFileName), wxsContent)
        File.WriteAllText(Path.Combine(outputDir, projectFileName), projectContent)
        File.WriteAllText(Path.Combine(outputDir, "build.cmd"), buildScript)

        {
            Project = project
            WxsContent = wxsContent
            ProjectFile = projectContent
            BuildScript = buildScript
            OutputDirectory = outputDir
            GeneratedFiles = [wxsFileName; projectFileName; "build.cmd"]
        }
