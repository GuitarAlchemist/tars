<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    <SuppressNETCoreSdkPreviewMessage>true</SuppressNETCoreSdkPreviewMessage>
  </PropertyGroup>

  <ItemGroup>
    <!-- Core Types -->
    <Compile Include="Core/Result.fs" />
    <Compile Include="Core/Option.fs" />
    <Compile Include="Core/AsyncResult.fs" />
    <Compile Include="Core/Collections.fs" />
    <Compile Include="Core/Interop.fs" />

    <!-- Consciousness Module -->
    <!-- Core -->
    <Compile Include="Consciousness/Core/Types.fs" />
    <Compile Include="Consciousness/Core/ConsciousnessCore.fs" />
    <Compile Include="Consciousness/Core/PureConsciousnessCore.fs" />
    <Compile Include="Consciousness/Services/IConsciousnessService.fs" />
    <Compile Include="Consciousness/Services/ConsciousnessService.fs" />
    <Compile Include="Consciousness/DependencyInjection/ServiceCollectionExtensions.fs" />
    
    <!-- Conceptual -->
    <Compile Include="Consciousness/Conceptual/Types.fs" />
    <Compile Include="Consciousness/Conceptual/Services/IConceptualService.fs" />
    <Compile Include="Consciousness/Conceptual/Services/ConceptualService.fs" />

    <!-- Decision -->
    <Compile Include="Consciousness\Decision\DependencyInjection\ServiceCollectionExtensionsNew.fs" />
    <Compile Include="Consciousness\Decision\Services\DecisionServiceComplete.fs" />
    <Compile Include="Consciousness\Decision\Services\DecisionServiceNew.fs" />
    <Compile Include="Consciousness\Decision\Services\DecisionServiceNew2.fs" />
    <Compile Include="Consciousness\Decision\Services\DecisionServiceNew3.fs" />
    <Compile Include="Consciousness\Decision\Services\DecisionServiceNew4.fs" />
    <Compile Include="Consciousness\Decision\Services\DecisionServiceUpdated.fs" />
    <Compile Include="Consciousness/Decision/Types.fs" />
    <Compile Include="Consciousness/Decision/Services/IDecisionService.fs" />
    <Compile Include="Consciousness/Decision/Services/DecisionService.fs" />
    <Compile Include="Consciousness/Decision/DependencyInjection/ServiceCollectionExtensions.fs" />

    <!-- Intelligence -->
    <Compile Include="Consciousness/Intelligence/Types.fs" />
    <Compile Include="Consciousness/Intelligence/Services/ICreativeThinking.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IIntuitiveReasoning.fs" />
    <Compile Include="Consciousness/Intelligence/Services/ISpontaneousThought.fs" />
    <Compile Include="Consciousness/Intelligence/Services/ICuriosityDrive.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IInsightGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IIntelligenceSpark.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CreativeThinking/CreativeThinkingBase.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CreativeThinking/CreativeIdeaGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CreativeThinking/CreativeSolutionGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CreativeThinking/CreativeThinking.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntuitiveReasoning/IntuitiveReasoningBase.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntuitiveReasoning/IntuitionGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntuitiveReasoning/IntuitiveDecisionMaking.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntuitiveReasoning/IntuitiveReasoning.fs" />
    <Compile Include="Consciousness/Intelligence/Services/SpontaneousThought/SpontaneousThoughtBase.fs" />
    <Compile Include="Consciousness/Intelligence/Services/SpontaneousThought/ThoughtGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/SpontaneousThought/SpontaneousThought.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CuriosityDrive/CuriosityDriveBase.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CuriosityDrive/QuestionGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CuriosityDrive/ExplorationMethods.fs" />
    <Compile Include="Consciousness/Intelligence/Services/CuriosityDrive/CuriosityDrive.fs" />
    <Compile Include="Consciousness/Intelligence/Services/InsightGeneration/InsightGenerationBase.fs" />
    <Compile Include="Consciousness/Intelligence/Services/InsightGeneration/ConnectionDiscovery.fs" />
    <Compile Include="Consciousness/Intelligence/Services/InsightGeneration/ProblemRestructuring.fs" />
    <Compile Include="Consciousness/Intelligence/Services/InsightGeneration/InsightGeneration.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntelligenceSpark/IntelligenceSparkBase.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntelligenceSpark/IntelligenceCoordination.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntelligenceSpark/IntelligenceReporting.fs" />
    <Compile Include="Consciousness/Intelligence/Services/IntelligenceSpark/IntelligenceSpark.fs" />
    <Compile Include="Consciousness/Intelligence/DependencyInjection/ServiceCollectionExtensions.fs" />

    <!-- Metascript -->
    <Compile Include="Metascript/Types.fs" />
    <Compile Include="Metascript/MetascriptExecutionResult.fs" />
    <Compile Include="Metascript/Services/IMetascriptService.fs" />
    <Compile Include="Metascript/Services/IMetascriptExecutor.fs" />
    <Compile Include="Metascript/Services/MetascriptService.fs" />
    <Compile Include="Metascript/Services/MetascriptExecutor.fs" />
    <Compile Include="Metascript/DependencyInjection/ServiceCollectionExtensions.fs" />

    <!-- ML -->
    <Compile Include="ML/Core/Types.fs" />
    <Compile Include="ML/Core/MLFrameworkOptions.fs" />
    <Compile Include="ML/Core/MLModelMetadata.fs" />
    <Compile Include="ML/Core/ModelMetadata.fs" />
    <Compile Include="ML/Core/MLException.fs" />
    <Compile Include="ML/Core/MLFramework.fs" />
    <Compile Include="ML/Services/IMLService.fs" />
    <Compile Include="ML/Services/MLService.fs" />
    <Compile Include="ML/DependencyInjection/ServiceCollectionExtensions.fs" />

    <!-- CodeAnalysis -->
    <Compile Include="CodeAnalysis/Types.fs" />
    <Compile Include="CodeAnalysis/CodeAnalyzer.fs" />

    <!-- CodeGen -->
    <Compile Include="CodeGen/Types.fs" />
    <Compile Include="CodeGen/Interfaces.fs" />
    <Compile Include="CodeGen/CodeGenerator.fs" />
    <Compile Include="CodeGen/Refactorer.fs" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="FSharp.Data" Version="6.3.0" />
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="Microsoft.Extensions.Logging.Abstractions" Version="8.0.0" />
    <PackageReference Include="Microsoft.Extensions.DependencyInjection.Abstractions" Version="8.0.0" />
    <PackageReference Include="Microsoft.ML" Version="3.0.0" />
  </ItemGroup>

</Project>
