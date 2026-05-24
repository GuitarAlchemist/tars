namespace TarsEngine.DSL

open System
open System.IO
open System.Security.Cryptography

/// <summary>
/// Module for handling privacy concerns with telemetry data.
/// </summary>
module TelemetryPrivacy =
    /// <summary>
    /// Anonymize a string by hashing it.
    /// </summary>
    /// <param name="input">The string to anonymize.</param>
    /// <returns>The anonymized string.</returns>
    let anonymizeString (input: string) =
        if String.IsNullOrEmpty(input) then
            ""
        else
            use sha256 = SHA256.Create()
            let hashBytes = sha256.ComputeHash(Text.Encoding.UTF8.GetBytes(input))
            BitConverter.ToString(hashBytes).Replace("-", "")
    
    /// <summary>
    /// Anonymize a file path by hashing each component.
    /// </summary>
    /// <param name="filePath">The file path to anonymize.</param>
    /// <returns>The anonymized file path.</returns>
    let anonymizeFilePath (filePath: string) =
        if String.IsNullOrEmpty(filePath) then
            ""
        else
            let directory = Path.GetDirectoryName(filePath)
            let fileName = Path.GetFileNameWithoutExtension(filePath)
            let extension = Path.GetExtension(filePath)
            
            let anonymizedDirectory = 
                if String.IsNullOrEmpty(directory) then
                    ""
                else
                    directory.Split([|Path.DirectorySeparatorChar; Path.AltDirectorySeparatorChar|], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.map anonymizeString
                    |> fun components -> String.Join(Path.DirectorySeparatorChar.ToString(), components)
            
            let anonymizedFileName = anonymizeString fileName
            
            Path.Combine(anonymizedDirectory, anonymizedFileName + extension)
    
    /// <summary>
    /// Anonymize telemetry data.
    /// </summary>
    /// <param name="telemetry">The telemetry data to anonymize.</param>
    /// <returns>The anonymized telemetry data.</returns>
    let anonymizeTelemetry (telemetry: TelemetryData) =
        // Create a new telemetry data object with anonymized values
        { telemetry with
            // Keep the ID, timestamp, parser version, operating system, and runtime version
            // Anonymize the usage telemetry
            UsageTelemetry = 
                { telemetry.UsageTelemetry with
                    // Keep the parser type, file size, line count, block count, property count, nested block count, and parse time
                    // The start and end timestamps are already UTC, so they don't need to be anonymized
                    StartTimestamp = telemetry.UsageTelemetry.StartTimestamp
                    EndTimestamp = telemetry.UsageTelemetry.EndTimestamp
                }
            
            // Anonymize the performance telemetry
            PerformanceTelemetry = 
                { telemetry.PerformanceTelemetry with
                    // Keep the parser type, file size, line count, and all timing information
                    // Keep the chunk count, cached chunk count, and peak memory usage
                }
            
            // Anonymize the error and warning telemetry
            ErrorWarningTelemetry = 
                telemetry.ErrorWarningTelemetry
                |> Option.map (fun ewt ->
                    { ewt with
                        // Keep the parser type, file size, line count, and all counts
                        // Keep the error codes, warning codes, and suppressed warning codes
                    }
                )
        }
    
    /// <summary>
    /// Purge telemetry data.
    /// </summary>
    /// <param name="directory">The directory containing telemetry data to purge.</param>
    /// <returns>The number of telemetry files purged.</returns>
    let purgeTelemetry (directory: string) =
        if Directory.Exists(directory) then
            let telemetryFiles = Directory.GetFiles(directory, "*.json")
            
            for file in telemetryFiles do
                try
                    File.Delete(file)
                with
                | _ -> ()
            
            telemetryFiles.Length
        else
            0
