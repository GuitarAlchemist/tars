using System.Diagnostics;

namespace TarsSupervisor.Cli.Utils
{
    public static class Proc
    {
        private static string Sanitize(string cmd)
        {
            if (string.IsNullOrWhiteSpace(cmd)) return cmd;
            cmd = cmd.Trim();

            // allow "powershell:" / "pwsh:" / "bash:" prefixes in config
            string[] prefixes = new[] { "powershell:", "pwsh:", "bash:" };
            foreach (var p in prefixes)
            {
                if (cmd.StartsWith(p, StringComparison.OrdinalIgnoreCase))
                    return cmd.Substring(p.Length).TrimStart();
            }
            return cmd;
        }

        public static async Task<(int code, string stdout, string stderr)> RunAsync(string cmd)
        {
            cmd = Sanitize(cmd);

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
            using var p = new Process { StartInfo = psi };
            p.Start();
            var stdout = await p.StandardOutput.ReadToEndAsync();
            var stderr = await p.StandardError.ReadToEndAsync();
            p.WaitForExit();
            return (p.ExitCode, stdout, stderr);
        }
    }
}
