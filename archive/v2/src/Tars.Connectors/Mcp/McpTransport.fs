namespace Tars.Connectors.Mcp

open System
open System.IO
open System.Threading.Tasks
open System.Diagnostics
open System.Text.Json

type IMcpTransport =
    abstract member StartAsync: unit -> Task
    abstract member SendAsync: JsonRpcRequest -> Task
    abstract member ReceiveAsync: unit -> Task<string option> // Returns raw JSON line
    abstract member CloseAsync: unit -> Task

type StdioTransport(command: string, arguments: string, workingDirectory: string option) =
    let mutable processOpt: Process option = None
    let mutable writer: StreamWriter option = None
    let mutable reader: StreamReader option = None

    interface IMcpTransport with
        member this.StartAsync() =
            task {
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- command
                startInfo.Arguments <- arguments
                startInfo.RedirectStandardInput <- true
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.UseShellExecute <- false
                startInfo.CreateNoWindow <- true

                match workingDirectory with
                | Some dir -> startInfo.WorkingDirectory <- dir
                | None -> ()

                let p = new Process()
                p.StartInfo <- startInfo

                p.ErrorDataReceived.Add(fun e ->
                    if not (String.IsNullOrEmpty(e.Data)) then
                        // In a real app, inject ILogger
                        Console.Error.WriteLine($"[MCP STDERR] {e.Data}"))

                if p.Start() then
                    p.BeginErrorReadLine()
                    processOpt <- Some p
                    writer <- Some p.StandardInput
                    reader <- Some p.StandardOutput
                else
                    failwith "Failed to start MCP process"
            }

        member this.SendAsync(request: JsonRpcRequest) =
            task {
                match writer with
                | Some w ->
                    let json = JsonSerializer.Serialize(request)
                    do! w.WriteLineAsync(json)
                    do! w.FlushAsync()
                | None -> failwith "Transport not started"
            }

        member this.ReceiveAsync() =
            task {
                match reader with
                | Some r ->
                    let! line = r.ReadLineAsync()
                    return if isNull line then None else Some line
                | None -> return None
            }

        member this.CloseAsync() =
            task {
                match processOpt with
                | Some p ->
                    if not p.HasExited then
                        p.Kill()

                    p.Dispose()
                | None -> ()
            }
