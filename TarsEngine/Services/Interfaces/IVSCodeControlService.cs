namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for controlling VS Code programmatically
/// </summary>
public interface IVSCodeControlService
{
    /// <summary>
    /// Execute a VS Code command
    /// </summary>
    /// <param name="command">The command to execute</param>
    /// <param name="args">Optional arguments for the command</param>
    /// <returns>True if the command was executed successfully, false otherwise</returns>
    Task<bool> ExecuteCommandAsync(string command, object args = null);
    
    /// <summary>
    /// Open a file in VS Code
    /// </summary>
    /// <param name="filePath">The path to the file</param>
    /// <returns>True if the file was opened successfully, false otherwise</returns>
    Task<bool> OpenFileAsync(string filePath);
    
    /// <summary>
    /// Insert text at the current cursor position
    /// </summary>
    /// <param name="text">The text to insert</param>
    /// <returns>True if the text was inserted successfully, false otherwise</returns>
    Task<bool> InsertTextAsync(string text);
    
    /// <summary>
    /// Select all text in the current editor
    /// </summary>
    /// <returns>True if the text was selected successfully, false otherwise</returns>
    Task<bool> SelectAllAsync();
    
    /// <summary>
    /// Save the current file
    /// </summary>
    /// <returns>True if the file was saved successfully, false otherwise</returns>
    Task<bool> SaveFileAsync();
    
    /// <summary>
    /// Open the command palette
    /// </summary>
    /// <returns>True if the command palette was opened successfully, false otherwise</returns>
    Task<bool> OpenCommandPaletteAsync();
    
    /// <summary>
    /// Execute a command via the command palette
    /// </summary>
    /// <param name="command">The command to execute</param>
    /// <returns>True if the command was executed successfully, false otherwise</returns>
    Task<bool> ExecuteCommandViaCommandPaletteAsync(string command);
}
