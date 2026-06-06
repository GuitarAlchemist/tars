namespace TarsEngine.Services;

public interface ITarsEngineService
{
    Task<ImprovementResult> GenerateImprovement(CancellationToken cancellationToken);
    Task<bool> LoadCheckpoint();
    Task SaveCheckpoint();
    Task<string> ProcessUploadedFile(Stream fileStream, string fileName);
    Task ProcessPrompt(string prompt);
}

public record ImprovementResult(
    string Capability,
    float Confidence,
    string Source
);