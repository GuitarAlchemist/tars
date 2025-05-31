using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Models;
using TarsCli.Services;

namespace TarsCli.Tests.Services;

public class ModelProviderFactoryTests
{
    private readonly Mock<ILogger<ModelProviderFactory>> _loggerMock;
    private readonly Mock<IConfiguration> _configurationMock;
    private readonly Mock<OllamaService> _ollamaServiceMock;
    private readonly Mock<DockerModelRunnerService> _dockerModelRunnerServiceMock;

    public ModelProviderFactoryTests()
    {
        _loggerMock = new Mock<ILogger<ModelProviderFactory>>();
        _configurationMock = new Mock<IConfiguration>();
        _ollamaServiceMock = new Mock<OllamaService>(
            MockBehavior.Loose,
            Mock.Of<ILogger<OllamaService>>(),
            Mock.Of<IConfiguration>(),
            Mock.Of<GpuService>());
        _dockerModelRunnerServiceMock = new Mock<DockerModelRunnerService>(
            MockBehavior.Loose,
            Mock.Of<ILogger<DockerModelRunnerService>>(),
            Mock.Of<IConfiguration>(),
            Mock.Of<GpuService>());
    }

    [Fact]
    public void Constructor_ShouldSetDefaultProvider_FromConfiguration()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("DockerModelRunner");
        _ollamaServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3");
        _dockerModelRunnerServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3:8b");

        // Act
        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Assert
        Assert.Equal(ModelProvider.DockerModelRunner, factory.DefaultProvider);
    }

    [Fact]
    public void Constructor_ShouldDefaultToOllama_WhenConfigurationMissing()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns((string)null);
        _ollamaServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3");
        _dockerModelRunnerServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3:8b");

        // Act
        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Assert
        Assert.Equal(ModelProvider.Ollama, factory.DefaultProvider);
    }

    [Fact]
    public async Task IsProviderAvailable_ShouldCheckOllama_WhenOllamaProviderSpecified()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("Ollama");
        _ollamaServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3");
        _ollamaServiceMock.Setup(x => x.IsModelAvailable("llama3")).ReturnsAsync(true);

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.IsProviderAvailable(ModelProvider.Ollama);

        // Assert
        Assert.True(result);
        _ollamaServiceMock.Verify(x => x.IsModelAvailable("llama3"), Times.Once);
    }

    [Fact]
    public async Task IsProviderAvailable_ShouldCheckDockerModelRunner_WhenDockerModelRunnerProviderSpecified()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("DockerModelRunner");
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable()).ReturnsAsync(true);

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.IsProviderAvailable(ModelProvider.DockerModelRunner);

        // Assert
        Assert.True(result);
        _dockerModelRunnerServiceMock.Verify(x => x.IsAvailable(), Times.Once);
    }

    [Fact]
    public async Task GenerateCompletion_ShouldUseOllama_WhenOllamaProviderSpecified()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("DockerModelRunner");
        _ollamaServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3");
        _ollamaServiceMock.Setup(x => x.GenerateCompletion("test prompt", "llama3"))
            .ReturnsAsync("Ollama response");

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GenerateCompletion("test prompt", "llama3", ModelProvider.Ollama);

        // Assert
        Assert.Equal("Ollama response", result);
        _ollamaServiceMock.Verify(x => x.GenerateCompletion("test prompt", "llama3"), Times.Once);
    }

    [Fact]
    public async Task GenerateCompletion_ShouldUseDockerModelRunner_WhenDockerModelRunnerProviderSpecified()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("Ollama");
        _dockerModelRunnerServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3:8b");
        _dockerModelRunnerServiceMock.Setup(x => x.GenerateCompletion("test prompt", "llama3:8b"))
            .ReturnsAsync("Docker Model Runner response");

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GenerateCompletion("test prompt", "llama3:8b", ModelProvider.DockerModelRunner);

        // Assert
        Assert.Equal("Docker Model Runner response", result);
        _dockerModelRunnerServiceMock.Verify(x => x.GenerateCompletion("test prompt", "llama3:8b"), Times.Once);
    }

    [Fact]
    public async Task GenerateCompletion_ShouldUseDefaultProvider_WhenNoProviderSpecified()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("Ollama");
        _ollamaServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3");
        _ollamaServiceMock.Setup(x => x.GenerateCompletion("test prompt", "llama3"))
            .ReturnsAsync("Ollama response");

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GenerateCompletion("test prompt");

        // Assert
        Assert.Equal("Ollama response", result);
        _ollamaServiceMock.Verify(x => x.GenerateCompletion("test prompt", "llama3"), Times.Once);
    }

    [Fact]
    public async Task GenerateCompletion_ShouldUseDefaultModel_WhenNoModelSpecified()
    {
        // Arrange
        _configurationMock.Setup(x => x["ModelProvider:Default"]).Returns("DockerModelRunner");
        _dockerModelRunnerServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3:8b");
        _dockerModelRunnerServiceMock.Setup(x => x.GenerateCompletion("test prompt", "llama3:8b"))
            .ReturnsAsync("Docker Model Runner response");

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GenerateCompletion("test prompt");

        // Assert
        Assert.Equal("Docker Model Runner response", result);
        _dockerModelRunnerServiceMock.Verify(x => x.GenerateCompletion("test prompt", "llama3:8b"), Times.Once);
    }

    [Fact]
    public async Task GetAvailableModelNames_ShouldReturnOllamaModels_WhenOllamaProviderSpecified()
    {
        // Arrange
        _ollamaServiceMock.Setup(x => x.GetAvailableModels())
            .ReturnsAsync(new List<string> { "llama3", "mistral" });

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GetAvailableModelNames(ModelProvider.Ollama);

        // Assert
        Assert.Equal(2, result.Count);
        Assert.Contains("llama3", result);
        Assert.Contains("mistral", result);
        _ollamaServiceMock.Verify(x => x.GetAvailableModels(), Times.Once);
    }

    [Fact]
    public async Task GetAvailableModelNames_ShouldReturnDockerModelRunnerModels_WhenDockerModelRunnerProviderSpecified()
    {
        // Arrange
        _dockerModelRunnerServiceMock.Setup(x => x.GetAvailableModels())
            .ReturnsAsync(new List<ModelInfo> 
            { 
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 },
                new ModelInfo { Id = "mistral:7b", OwnedBy = "mistral", Created = 1717171717 }
            });

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GetAvailableModelNames(ModelProvider.DockerModelRunner);

        // Assert
        Assert.Equal(2, result.Count);
        Assert.Contains("llama3:8b", result);
        Assert.Contains("mistral:7b", result);
        _dockerModelRunnerServiceMock.Verify(x => x.GetAvailableModels(), Times.Once);
    }

    [Fact]
    public async Task IsModelAvailable_ShouldCheckOllama_WhenOllamaProviderSpecified()
    {
        // Arrange
        _ollamaServiceMock.Setup(x => x.IsModelAvailable("llama3"))
            .ReturnsAsync(true);

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.IsModelAvailable("llama3", ModelProvider.Ollama);

        // Assert
        Assert.True(result);
        _ollamaServiceMock.Verify(x => x.IsModelAvailable("llama3"), Times.Once);
    }

    [Fact]
    public async Task IsModelAvailable_ShouldCheckDockerModelRunner_WhenDockerModelRunnerProviderSpecified()
    {
        // Arrange
        _dockerModelRunnerServiceMock.Setup(x => x.IsModelAvailable("llama3:8b"))
            .ReturnsAsync(true);

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.IsModelAvailable("llama3:8b", ModelProvider.DockerModelRunner);

        // Assert
        Assert.True(result);
        _dockerModelRunnerServiceMock.Verify(x => x.IsModelAvailable("llama3:8b"), Times.Once);
    }

    [Fact]
    public async Task GenerateChatCompletion_ShouldUseDockerModelRunner_WhenDockerModelRunnerProviderSpecified()
    {
        // Arrange
        var messages = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "Hello" }
        };

        _dockerModelRunnerServiceMock.Setup(x => x.GenerateChatCompletion(messages, "llama3:8b"))
            .ReturnsAsync("Docker Model Runner response");

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GenerateChatCompletion(messages, "llama3:8b", ModelProvider.DockerModelRunner);

        // Assert
        Assert.Equal("Docker Model Runner response", result);
        _dockerModelRunnerServiceMock.Verify(x => x.GenerateChatCompletion(messages, "llama3:8b"), Times.Once);
    }

    [Fact]
    public async Task GenerateChatCompletion_ShouldConvertToPrompt_WhenOllamaProviderSpecified()
    {
        // Arrange
        var messages = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "Hello" },
            new ChatMessage { Role = "assistant", Content = "Hi there" },
            new ChatMessage { Role = "user", Content = "How are you?" }
        };

        _ollamaServiceMock.Setup(x => x.GenerateCompletion(It.IsAny<string>(), "llama3"))
            .ReturnsAsync("Ollama response");

        var factory = new ModelProviderFactory(
            _loggerMock.Object,
            _configurationMock.Object,
            _ollamaServiceMock.Object,
            _dockerModelRunnerServiceMock.Object);

        // Act
        var result = await factory.GenerateChatCompletion(messages, "llama3", ModelProvider.Ollama);

        // Assert
        Assert.Equal("Ollama response", result);
        _ollamaServiceMock.Verify(x => x.GenerateCompletion(
            "user: Hello\n\nassistant: Hi there\n\nuser: How are you?", 
            "llama3"), 
            Times.Once);
    }
}
