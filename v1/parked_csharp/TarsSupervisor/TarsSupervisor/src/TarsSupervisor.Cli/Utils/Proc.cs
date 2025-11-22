using System.Diagnostics;

namespace TarsSupervisor.Cli.Utils;

public static class Proc
{
    public static async Task<(int code, string stdout, string stderr)> RunAsync(string cmd)
    {
        bool isWindows = OperatingSystem.IsWindows();
        var psi = new ProcessStartInfo
        {
            FileName = isWindows ? "powershell" : "/bin/bash",
            ArgumentList = { isWindows ? "-NoProfile" : "-lc", isWindows ? cmd : cmd },
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        var p = new Process { StartInfo = psi };
        p.Start();
        var stdout = await p.StandardOutput.ReadToEndAsync();
        var stderr = await p.StandardError.ReadToEndAsync();
        p.WaitForExit();
        return (p.ExitCode, stdout, stderr);
    }
}
