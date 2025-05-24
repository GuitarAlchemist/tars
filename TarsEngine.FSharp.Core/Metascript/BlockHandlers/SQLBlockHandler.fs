namespace TarsEngine.FSharp.Core.Metascript.BlockHandlers

open System
open System.Data
open System.Data.SqlClient
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript

/// <summary>
/// Handler for SQL blocks.
/// </summary>
type SQLBlockHandler(logger: ILogger<SQLBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.SQL, 60)
    
    /// <summary>
    /// Executes SQL code.
    /// </summary>
    /// <param name="code">The code to execute.</param>
    /// <param name="connectionString">The connection string.</param>
    /// <returns>The output, error, and success status.</returns>
    let executeSQLCode (code: string) (connectionString: string) =
        try
            use connection = new SqlConnection(connectionString)
            connection.Open()
            
            use command = new SqlCommand(code, connection)
            
            // Check if the command is a query
            if code.Trim().ToUpperInvariant().StartsWith("SELECT") then
                // Execute a query
                use reader = command.ExecuteReader()
                
                // Build the output
                let sb = StringBuilder()
                
                // Get the column names
                let columnCount = reader.FieldCount
                let columnNames = [| for i in 0 .. columnCount - 1 -> reader.GetName(i) |]
                
                // Write the header
                sb.AppendLine(String.Join("\t", columnNames)) |> ignore
                
                // Write the data
                while reader.Read() do
                    let values = [| for i in 0 .. columnCount - 1 -> if reader.IsDBNull(i) then "NULL" else reader.GetValue(i).ToString() |]
                    sb.AppendLine(String.Join("\t", values)) |> ignore
                
                (sb.ToString(), "", true)
            else
                // Execute a non-query
                let rowsAffected = command.ExecuteNonQuery()
                ($"{rowsAffected} rows affected", "", true)
        with
        | ex ->
            ("", ex.ToString(), false)
    
    /// <summary>
    /// Executes a SQL block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                // Get the connection string from parameters or context
                let connectionString = 
                    block.Parameters
                    |> List.tryFind (fun p -> p.Name = "connection" || p.Name = "connectionString")
                    |> Option.map (fun p -> p.Value)
                    |> Option.defaultWith (fun () ->
                        context.Variables
                        |> Map.tryFind "connectionString"
                        |> Option.map (fun v -> v.Value.ToString())
                        |> Option.defaultValue "")
                
                // Check if we have a connection string
                if String.IsNullOrWhiteSpace(connectionString) then
                    return this.CreateFailureResult(block, "No connection string provided")
                else
                    // Execute the SQL code
                    let (output, error, success) = executeSQLCode block.Content connectionString
                    
                    // Check for errors
                    if not success then
                        return this.CreateFailureResult(block, error, output)
                    else
                        return this.CreateSuccessResult(block, output)
            with
            | ex ->
                logger.LogError(ex, "Error executing SQL block")
                return this.CreateFailureResult(block, ex.ToString())
        }
