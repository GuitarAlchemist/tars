namespace TarsCli.Constants;

/// <summary>
/// Docker Model Runner API endpoints
/// </summary>
public static class DockerModelRunnerEndpoints
{
    /// <summary>
    /// Base API path
    /// </summary>
    public const string ApiBase = "/v1";

    /// <summary>
    /// Models endpoint
    /// </summary>
    public const string Models = ApiBase + "/models";

    /// <summary>
    /// Completions endpoint
    /// </summary>
    public const string Completions = ApiBase + "/completions";

    /// <summary>
    /// Chat completions endpoint
    /// </summary>
    public const string ChatCompletions = ApiBase + "/chat/completions";
}
