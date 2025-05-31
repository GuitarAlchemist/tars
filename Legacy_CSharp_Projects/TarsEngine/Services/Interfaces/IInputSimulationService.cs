namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for simulating keyboard and mouse input
/// </summary>
public interface IInputSimulationService
{
    /// <summary>
    /// Type text using simulated keyboard input
    /// </summary>
    /// <param name="text">The text to type</param>
    /// <returns>A task representing the asynchronous operation</returns>
    Task TypeTextAsync(string text);
    
    /// <summary>
    /// Press keys using simulated keyboard input
    /// </summary>
    /// <param name="keys">The keys to press (in SendKeys format)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    Task PressKeysAsync(string keys);
    
    /// <summary>
    /// Move the mouse to a specific position
    /// </summary>
    /// <param name="x">The x-coordinate</param>
    /// <param name="y">The y-coordinate</param>
    /// <returns>A task representing the asynchronous operation</returns>
    Task MoveMouseAsync(int x, int y);
    
    /// <summary>
    /// Click the mouse at the current position
    /// </summary>
    /// <param name="button">The mouse button to click (left, right, middle)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    Task ClickMouseAsync(string button = "left");
    
    /// <summary>
    /// Move the mouse to a specific position and click
    /// </summary>
    /// <param name="x">The x-coordinate</param>
    /// <param name="y">The y-coordinate</param>
    /// <param name="button">The mouse button to click (left, right, middle)</param>
    /// <returns>A task representing the asynchronous operation</returns>
    Task MoveAndClickAsync(int x, int y, string button = "left");
}
