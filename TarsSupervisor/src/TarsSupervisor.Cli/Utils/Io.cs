using System.Text;
namespace TarsSupervisor.Cli.Utils;

public static class Io
{
    public static void EnsureDir(string path) => Directory.CreateDirectory(path);
    public static string SlugTimeUtc() => DateTimeOffset.UtcNow.ToString("yyyyMMdd_HHmmss");

    public static async Task WriteAllTextAsync(string path, string content)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir!);
        await File.WriteAllTextAsync(path, content, Encoding.UTF8);
    }

    public static async Task AppendAllTextAsync(string path, string content)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir!);
        await File.AppendAllTextAsync(path, content, Encoding.UTF8);
    }
}
