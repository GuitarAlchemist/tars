using System.Net;
using System.Net.Http;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Moq;
using Moq.Protected;
using TarsCli.Constants;
using TarsCli.Models;
using TarsCli.Services;

namespace TarsCli.Tests.Services;

public class DockerModelRunnerServiceTests
{
    private readonly Mock<ILogger<DockerModelRunnerService>> _loggerMock;
    private readonly Mock<IConfiguration> _configurationMock;
    private readonly Mock<GpuService> _gpuServiceMock;
    private readonly Mock<HttpMessageHandler> _httpMessageHandlerMock;
    private readonly HttpClient _httpClient;

    public DockerModelRunnerServiceTests()
    {
        _loggerMock = new Mock<ILogger<DockerModelRunnerService>>();
        _configurationMock = new Mock<IConfiguration>();
        _gpuServiceMock = new Mock<GpuService>(
            MockBehavior.Loose,
            Mock.Of<ILogger<GpuService>>(),
            Mock.Of<IConfiguration>());
        _httpMessageHandlerMock = new Mock<HttpMessageHandler>();
        _httpClient = new HttpClient(_httpMessageHandlerMock.Object);
    }

    private DockerModelRunnerService CreateService()
    {
        _configurationMock.Setup(x => x[ConfigurationKeys.DockerModelRunner.BaseUrl])
            .Returns("http://localhost:8080");
        _configurationMock.Setup(x => x[ConfigurationKeys.DockerModelRunner.DefaultModel])
            .Returns("llama3:8b");
        _gpuServiceMock.Setup(x => x.IsGpuAvailable())
            .Returns(true);

        return new DockerModelRunnerService(
            _loggerMock.Object,
            _configurationMock.Object,
            _gpuServiceMock.Object);
    }

    [Fact]
    public void Constructor_ShouldSetBaseUrlAndDefaultModel_FromConfiguration()
    {
        // Arrange
        _configurationMock.Setup(x => x[ConfigurationKeys.DockerModelRunner.BaseUrl])
            .Returns("http://custom:8080");
        _configurationMock.Setup(x => x[ConfigurationKeys.DockerModelRunner.DefaultModel])
            .Returns("custom-model");
        _gpuServiceMock.Setup(x => x.IsGpuAvailable())
            .Returns(true);

        // Act
        var service = new DockerModelRunnerService(
            _loggerMock.Object,
            _configurationMock.Object,
            _gpuServiceMock.Object);

        // Assert
        Assert.Equal("http://custom:8080", service.BaseUrl);
        Assert.Equal("custom-model", service.DefaultModel);
    }

    [Fact]
    public void Constructor_ShouldUseDefaultValues_WhenConfigurationMissing()
    {
        // Arrange
        _configurationMock.Setup(x => x[ConfigurationKeys.DockerModelRunner.BaseUrl])
            .Returns((string)null);
        _configurationMock.Setup(x => x[ConfigurationKeys.DockerModelRunner.DefaultModel])
            .Returns((string)null);
        _gpuServiceMock.Setup(x => x.IsGpuAvailable())
            .Returns(false);

        // Act
        var service = new DockerModelRunnerService(
            _loggerMock.Object,
            _configurationMock.Object,
            _gpuServiceMock.Object);

        // Assert
        Assert.Equal("http://localhost:8080", service.BaseUrl);
        Assert.Equal("llama3:8b", service.DefaultModel);
    }

    [Fact]
    public async Task IsAvailable_ShouldReturnTrue_WhenApiReturnsSuccess()
    {
        // Arrange
        var service = CreateService();

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent("{\"models\": []}")
            });

        // Act
        var result = await service.IsAvailable();

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task IsAvailable_ShouldReturnFalse_WhenApiReturnsError()
    {
        // Arrange
        var service = CreateService();

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError
            });

        // Act
        var result = await service.IsAvailable();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task IsModelAvailable_ShouldReturnTrue_WhenModelExists()
    {
        // Arrange
        var service = CreateService();
        var modelListResponse = new ModelListResponse
        {
            Models = new List<ModelInfo>
            {
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 },
                new ModelInfo { Id = "mistral:7b", OwnedBy = "mistral", Created = 1717171717 }
            }
        };

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(modelListResponse))
            });

        // Act
        var result = await service.IsModelAvailable("llama3:8b");

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task IsModelAvailable_ShouldReturnFalse_WhenModelDoesNotExist()
    {
        // Arrange
        var service = CreateService();
        var modelListResponse = new ModelListResponse
        {
            Models = new List<ModelInfo>
            {
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 }
            }
        };

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(modelListResponse))
            });

        // Act
        var result = await service.IsModelAvailable("nonexistent-model");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task GetAvailableModels_ShouldReturnModels_WhenApiReturnsSuccess()
    {
        // Arrange
        var service = CreateService();
        var modelListResponse = new ModelListResponse
        {
            Models = new List<ModelInfo>
            {
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 },
                new ModelInfo { Id = "mistral:7b", OwnedBy = "mistral", Created = 1717171717 }
            }
        };

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(modelListResponse))
            });

        // Act
        var result = await service.GetAvailableModels();

        // Assert
        Assert.Equal(2, result.Count);
        Assert.Equal("llama3:8b", result[0].Id);
        Assert.Equal("mistral:7b", result[1].Id);
    }

    [Fact]
    public async Task GetAvailableModels_ShouldReturnEmptyList_WhenApiReturnsError()
    {
        // Arrange
        var service = CreateService();

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError
            });

        // Act
        var result = await service.GetAvailableModels();

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public async Task GenerateCompletion_ShouldReturnCompletion_WhenApiReturnsSuccess()
    {
        // Arrange
        var service = CreateService();
        var completionResponse = new CompletionResponse
        {
            Id = "cmpl-123",
            Object = "completion",
            Created = 1717171717,
            Model = "llama3:8b",
            Choices = new List<CompletionChoice>
            {
                new CompletionChoice
                {
                    Text = "This is a test completion",
                    Index = 0,
                    FinishReason = "stop"
                }
            }
        };

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(completionResponse))
            });

        // Act
        var result = await service.GenerateCompletion("test prompt", "llama3:8b");

        // Assert
        Assert.Equal("This is a test completion", result);
    }

    [Fact]
    public async Task GenerateCompletion_ShouldReturnErrorMessage_WhenApiReturnsError()
    {
        // Arrange
        var service = CreateService();

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError,
                ReasonPhrase = "Internal Server Error"
            });

        // Act
        var result = await service.GenerateCompletion("test prompt", "llama3:8b");

        // Assert
        Assert.Contains("Error:", result);
    }

    [Fact]
    public async Task GenerateChatCompletion_ShouldReturnCompletion_WhenApiReturnsSuccess()
    {
        // Arrange
        var service = CreateService();
        var chatCompletionResponse = new ChatCompletionResponse
        {
            Id = "cmpl-123",
            Object = "chat.completion",
            Created = 1717171717,
            Model = "llama3:8b",
            Choices = new List<ChatCompletionChoice>
            {
                new ChatCompletionChoice
                {
                    Index = 0,
                    Message = new ChatMessageModel
                    {
                        Role = "assistant",
                        Content = "This is a test chat completion"
                    },
                    FinishReason = "stop"
                }
            }
        };

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = new StringContent(JsonSerializer.Serialize(chatCompletionResponse))
            });

        var messages = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "Hello" }
        };

        // Act
        var result = await service.GenerateChatCompletion(messages, "llama3:8b");

        // Assert
        Assert.Equal("This is a test chat completion", result);
    }

    [Fact]
    public async Task GenerateChatCompletion_ShouldReturnErrorMessage_WhenApiReturnsError()
    {
        // Arrange
        var service = CreateService();

        _httpMessageHandlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.InternalServerError,
                ReasonPhrase = "Internal Server Error"
            });

        var messages = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "Hello" }
        };

        // Act
        var result = await service.GenerateChatCompletion(messages, "llama3:8b");

        // Assert
        Assert.Contains("Error:", result);
    }

    [Fact]
    public async Task PullModel_ShouldReturnTrue_WhenSuccessful()
    {
        // Arrange
        var service = CreateService();

        // Act
        var result = await service.PullModel("llama3:8b");

        // Assert
        Assert.True(result);
    }
}
