namespace TarsEngine;

public interface ITarsEngine
{
    Task<(DateTime Time, string Capability, float Confidence)> GenerateImprovement();
    Task ResumeLastSession();
}