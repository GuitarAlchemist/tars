using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for simulating keyboard and mouse input
/// </summary>
public class InputSimulationService : IInputSimulationService
{
    private readonly ILogger<InputSimulationService> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="InputSimulationService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public InputSimulationService(ILogger<InputSimulationService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Type text using simulated keyboard input
    /// </summary>
    /// <param name="text">The text to type</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task TypeTextAsync(string text)
    {
        try
        {
            _logger.LogInformation($"Typing text: {text.Substring(0, Math.Min(text.Length, 50))}...");
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // On Windows, use SendKeys via PowerShell
                await SendKeysViaPowerShellAsync(text);
            }
            else
            {
                // On other platforms, use xdotool
                await SendKeysViaXdotoolAsync(text);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error typing text");
            throw;
        }
    }

    /// <summary>
    /// Press keys using simulated keyboard input
    /// </summary>
    /// <param name="keys">The keys to press (in SendKeys format)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task PressKeysAsync(string keys)
    {
        try
        {
            _logger.LogInformation($"Pressing keys: {keys}");
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // On Windows, use SendKeys via PowerShell
                await SendKeysViaPowerShellAsync(keys);
            }
            else
            {
                // On other platforms, use xdotool
                await SendKeysViaXdotoolAsync(ConvertToXdotoolFormat(keys));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error pressing keys: {keys}");
            throw;
        }
    }

    /// <summary>
    /// Move the mouse to a specific position
    /// </summary>
    /// <param name="x">The x-coordinate</param>
    /// <param name="y">The y-coordinate</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task MoveMouseAsync(int x, int y)
    {
        try
        {
            _logger.LogInformation($"Moving mouse to position: ({x}, {y})");
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // On Windows, use PowerShell
                await MoveMouseViaPowerShellAsync(x, y);
            }
            else
            {
                // On other platforms, use xdotool
                await MoveMouseViaXdotoolAsync(x, y);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error moving mouse to position: ({x}, {y})");
            throw;
        }
    }

    /// <summary>
    /// Click the mouse at the current position
    /// </summary>
    /// <param name="button">The mouse button to click (left, right, middle)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task ClickMouseAsync(string button = "left")
    {
        try
        {
            _logger.LogInformation($"Clicking mouse button: {button}");
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // On Windows, use PowerShell
                await ClickMouseViaPowerShellAsync(button);
            }
            else
            {
                // On other platforms, use xdotool
                await ClickMouseViaXdotoolAsync(button);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error clicking mouse button: {button}");
            throw;
        }
    }

    /// <summary>
    /// Move the mouse to a specific position and click
    /// </summary>
    /// <param name="x">The x-coordinate</param>
    /// <param name="y">The y-coordinate</param>
    /// <param name="button">The mouse button to click (left, right, middle)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task MoveAndClickAsync(int x, int y, string button = "left")
    {
        try
        {
            _logger.LogInformation($"Moving mouse to position ({x}, {y}) and clicking {button} button");
            
            // Move the mouse
            await MoveMouseAsync(x, y);
            
            // Wait a moment for the move to complete
            await Task.Delay(100);
            
            // Click the mouse
            await ClickMouseAsync(button);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error moving and clicking mouse at position: ({x}, {y})");
            throw;
        }
    }

    /// <summary>
    /// Send keys via PowerShell on Windows
    /// </summary>
    /// <param name="keys">The keys to send</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task SendKeysViaPowerShellAsync(string keys)
    {
        // Escape single quotes in the keys string
        var escapedKeys = keys.Replace("'", "''");
        
        // Create the PowerShell script
        var script = $@"
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.SendKeys]::SendWait('{escapedKeys}')
        ";
        
        // Execute the PowerShell script
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-Command \"{script}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        await process.WaitForExitAsync();
    }

    /// <summary>
    /// Send keys via xdotool on Linux
    /// </summary>
    /// <param name="keys">The keys to send</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task SendKeysViaXdotoolAsync(string keys)
    {
        // Create the xdotool command
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "xdotool",
                Arguments = $"type \"{keys}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        await process.WaitForExitAsync();
    }

    /// <summary>
    /// Move the mouse via PowerShell on Windows
    /// </summary>
    /// <param name="x">The x-coordinate</param>
    /// <param name="y">The y-coordinate</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task MoveMouseViaPowerShellAsync(int x, int y)
    {
        // Create the PowerShell script
        var script = $@"
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({x}, {y})
        ";
        
        // Execute the PowerShell script
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-Command \"{script}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        await process.WaitForExitAsync();
    }

    /// <summary>
    /// Move the mouse via xdotool on Linux
    /// </summary>
    /// <param name="x">The x-coordinate</param>
    /// <param name="y">The y-coordinate</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task MoveMouseViaXdotoolAsync(int x, int y)
    {
        // Create the xdotool command
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "xdotool",
                Arguments = $"mousemove {x} {y}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        await process.WaitForExitAsync();
    }

    /// <summary>
    /// Click the mouse via PowerShell on Windows
    /// </summary>
    /// <param name="button">The mouse button to click (left, right, middle)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task ClickMouseViaPowerShellAsync(string button)
    {
        // Create the PowerShell script
        var script = $@"
            Add-Type -AssemblyName System.Windows.Forms
            $signature = @""
                [DllImport(""user32.dll"", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
                public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint cButtons, uint dwExtraInfo);
            ""@
            $type = Add-Type -MemberDefinition $signature -Name ""MouseEvents"" -Namespace ""Win32"" -PassThru
            
            # Constants for mouse events
            $MOUSEEVENTF_LEFTDOWN = 0x02
            $MOUSEEVENTF_LEFTUP = 0x04
            $MOUSEEVENTF_RIGHTDOWN = 0x08
            $MOUSEEVENTF_RIGHTUP = 0x10
            $MOUSEEVENTF_MIDDLEDOWN = 0x20
            $MOUSEEVENTF_MIDDLEUP = 0x40
            
            # Perform the click based on the button
            switch (""{button.ToLower()}"") {{
                ""left"" {{
                    $type::mouse_event($MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    Start-Sleep -Milliseconds 10
                    $type::mouse_event($MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                }}
                ""right"" {{
                    $type::mouse_event($MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    Start-Sleep -Milliseconds 10
                    $type::mouse_event($MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                }}
                ""middle"" {{
                    $type::mouse_event($MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                    Start-Sleep -Milliseconds 10
                    $type::mouse_event($MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                }}
            }}
        ";
        
        // Execute the PowerShell script
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-Command \"{script}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        await process.WaitForExitAsync();
    }

    /// <summary>
    /// Click the mouse via xdotool on Linux
    /// </summary>
    /// <param name="button">The mouse button to click (left, right, middle)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task ClickMouseViaXdotoolAsync(string button)
    {
        // Map button names to xdotool button numbers
        var buttonNumber = button.ToLower() switch
        {
            "left" => 1,
            "middle" => 2,
            "right" => 3,
            _ => 1 // Default to left button
        };
        
        // Create the xdotool command
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "xdotool",
                Arguments = $"click {buttonNumber}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            }
        };
        
        process.Start();
        await process.WaitForExitAsync();
    }

    /// <summary>
    /// Convert SendKeys format to xdotool format
    /// </summary>
    /// <param name="keys">The keys in SendKeys format</param>
    /// <returns>The keys in xdotool format</returns>
    private string ConvertToXdotoolFormat(string keys)
    {
        // This is a simplified conversion and may not handle all cases
        var result = keys;
        
        // Convert special keys
        result = result.Replace("^", "ctrl+");
        result = result.Replace("+", "shift+");
        result = result.Replace("%", "alt+");
        
        // Convert braces
        result = result.Replace("{ENTER}", "Return");
        result = result.Replace("{ESC}", "Escape");
        result = result.Replace("{TAB}", "Tab");
        result = result.Replace("{BACKSPACE}", "BackSpace");
        result = result.Replace("{DELETE}", "Delete");
        result = result.Replace("{HOME}", "Home");
        result = result.Replace("{END}", "End");
        result = result.Replace("{PGUP}", "Page_Up");
        result = result.Replace("{PGDN}", "Page_Down");
        
        return result;
    }
}
