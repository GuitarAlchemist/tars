﻿# Script to fix the TarsEngineFSharp.fsproj file

$content = @"
<Project Sdk="Microsoft.NET.Sdk">

    <PropertyGroup>
        <TargetFramework>net9.0</TargetFramework>
        <GenerateDocumentationFile>true</GenerateDocumentationFile>
        <SuppressNETCoreSdkPreviewMessage>true</SuppressNETCoreSdkPreviewMessage>
    </PropertyGroup>

    <ItemGroup>
        <Compile Include="TreeOfThought\ThoughtNode.fs" />
        <Compile Include="TreeOfThought\ThoughtTree.fs" />
        <Compile Include="TreeOfThought\Evaluation.fs" />
        <Compile Include="Agents\AgentTypes.fs" />
        <Compile Include="RetroactionAnalysis.fs" />
        <Compile Include="JavaScriptAnalysis.fs" />
        <Compile Include="Agents\AnalysisAgent.fs" />
        <Compile Include="Agents\ValidationAgent.fs" />
        <Compile Include="Agents\TransformationAgent.fs" />
        <Compile Include="CodeAnalysis.fs" />
        <Compile Include="EnhancedCodeAnalysis.fs" />
        <Compile Include="MetascriptEngine.fs" />
        <Compile Include="Option.fs" />
        <Compile Include="Result.fs" />
        <Compile Include="ModelProvider.fs" />
        <Compile Include="AsyncExecution.fs" />
        <Compile Include="DataProcessing.fs" />
        <Compile Include="DataSources.fs" />
        <Compile Include="PromptEngine.fs" />
        <Compile Include="SampleAgents.fs" />
        <Compile Include="TarsDsl.fs" />
        <Compile Include="Examples.fs" />
        <Compile Include="RivaService.fs" />
        <Compile Include="TypeProviderPatternMatching.fs" />
        <Compile Include="WeatherService.fs" />
        <Compile Include="ChatService.fs" />
        <Compile Include="LlmService.fs" />
    </ItemGroup>

    <ItemGroup>
        <PackageReference Include="FSharp.Data" Version="6.3.0" />
        <PackageReference Include="Microsoft.CodeAnalysis.CSharp" Version="4.8.0" />
        <PackageReference Include="FSharp.TypeProviders.SDK" Version="7.0.2" />
        <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
        <PackageReference Include="Grpc.Core" Version="2.46.6" />
        <PackageReference Include="Google.Protobuf" Version="3.25.1" />
        <PackageReference Include="Grpc.Net.Client" Version="2.59.0" />
    </ItemGroup>

    <ItemGroup>
      <ProjectReference Include="..\TarsEngine.Interfaces\TarsEngine.Interfaces.csproj" />
    </ItemGroup>

    <ItemGroup>
      <Content Include="improvement_rules.meta" />
    </ItemGroup>

</Project>
"@

# Write the fixed content to the file
Set-Content -Path "TarsEngineFSharp\TarsEngineFSharp.fsproj" -Value $content
