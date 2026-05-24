namespace Tars.Interface.Cli

open System
open System.Net
open System.Threading
open System.Threading.Tasks
open System.Text.Json
open System.Collections.Generic
open Tars.Core
open Tars.Knowledge

module InfrastructureServer =

    let mutable private ledgerProvider : (unit -> Task<KnowledgeLedger option>) = fun () -> Task.FromResult(None)

    let setLedgerProvider provider = ledgerProvider <- provider

    type GraphNode = { id: string; group: int; label: string }
    type GraphLink = { source: string; target: string; label: string; value: int }
    type GraphData = { nodes: GraphNode list; links: GraphLink list }

    let start (port: int) (health: HealthCheckRegistry) (cancellationToken: CancellationToken) =
        let tryStart prefixes =
            try
                let listener = new HttpListener()
                for p in prefixes do listener.Prefixes.Add(p)
                listener.Start()
                Some listener
            with _ ->
                None

        let listener = 
            // Try wildcard first
            match tryStart [ $"http://*:{port}/" ] with
            | Some l -> 
                Console.WriteLine($"[Infra] Serving metrics, health & graph on http://*:{port}/")
                Some l
            | None ->
                // Fallback to localhost
                match tryStart [ $"http://localhost:{port}/" ] with
                | Some l ->
                    Console.WriteLine($"[Infra] Serving metrics, health & graph on http://localhost:{port}/")
                    Some l
                | None ->
                    Console.WriteLine("[Infra] Failed to start metrics server on both wildcard and localhost.")
                    None

        match listener with
        | Some listener ->
            Task.Run(fun () ->
                while not cancellationToken.IsCancellationRequested && listener.IsListening do
                    try
                        let contextTask = listener.GetContextAsync()
                        contextTask.Wait(cancellationToken)
                        let context = contextTask.Result
                        let response = context.Response
                        
                        match context.Request.Url.AbsolutePath with
                        | "/health" ->
                            let healthTask = health.RunAllAsync()
                            healthTask.Wait()
                            let json = HealthChecks.formatJson healthTask.Result
                            let buffer = System.Text.Encoding.UTF8.GetBytes(json)
                            response.ContentType <- "application/json"
                            response.ContentLength64 <- int64 buffer.Length
                            response.OutputStream.Write(buffer, 0, buffer.Length)
                            response.StatusCode <- if healthTask.Result.Status = Healthy then 200 else 503

                        | "/metrics" ->
                            let report = Metrics.toPrometheus()
                            let buffer = System.Text.Encoding.UTF8.GetBytes(report)
                            response.ContentType <- "text/plain; version=0.0.4"
                            response.ContentLength64 <- int64 buffer.Length
                            response.OutputStream.Write(buffer, 0, buffer.Length)
                            response.StatusCode <- 200

                        | "/graph" ->
                            let html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TARS Knowledge Graph</title>
    <script src="//unpkg.com/3d-force-graph"></script>
    <style> 
        body { margin: 0; } 
        #info { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); color: white; padding: 10px; font-family: monospace; pointer-events: none; }
    </style>
</head>
<body>
    <div id="info">Hover over a node</div>
    <div id="3d-graph"></div>
    <script>
        const elem = document.getElementById('3d-graph');
        const info = document.getElementById('info');
        
        const Graph = ForceGraph3D()
            (elem)
            .jsonUrl('/graph/data')
            .nodeLabel('label')
            .nodeAutoColorBy('group')
            .linkDirectionalArrowLength(3.5)
            .linkDirectionalArrowRelPos(1)
            .linkLabel(l => `${l.label}`)
            .onNodeHover(node => {
                elem.style.cursor = node ? 'pointer' : null;
                info.innerText = node ? `Node: ${node.label} (${node.group})` : 'Hover over a node';
            });
    </script>
</body>
</html>"""
                            let buffer = System.Text.Encoding.UTF8.GetBytes(html)
                            response.ContentType <- "text/html"
                            response.ContentLength64 <- int64 buffer.Length
                            response.OutputStream.Write(buffer, 0, buffer.Length)
                            response.StatusCode <- 200

                        | "/graph/data" ->
                            // Async wait for ledger
                            let ledgerTask = ledgerProvider()
                            ledgerTask.Wait()
                            
                            match ledgerTask.Result with
                            | Some ledger ->
                                let beliefs = ledger.Query() |> Seq.toList
                                
                                let nodes = HashSet<string>()
                                let links = ResizeArray<GraphLink>()
                                
                                for b in beliefs do
                                    nodes.Add(b.Subject.Value) |> ignore
                                    nodes.Add(b.Object.Value) |> ignore
                                    links.Add({ source = b.Subject.Value; target = b.Object.Value; label = b.Predicate.ToString(); value = 1 })
                                
                                let graphNodes = 
                                    nodes 
                                    |> Seq.map (fun n -> { id = n; group = 1; label = n })
                                    |> Seq.toList
                                    
                                let graphData = { nodes = graphNodes; links = links |> Seq.toList }
                                let json = JsonSerializer.Serialize(graphData)
                                let buffer = System.Text.Encoding.UTF8.GetBytes(json)
                                
                                response.ContentType <- "application/json"
                                response.ContentLength64 <- int64 buffer.Length
                                response.OutputStream.Write(buffer, 0, buffer.Length)
                                response.StatusCode <- 200
                            | None ->
                                let msg = "Ledger not initialized. Run 'tars service' or ensure the command initializes the ledger."
                                let buffer = System.Text.Encoding.UTF8.GetBytes(msg)
                                response.StatusCode <- 503
                                response.OutputStream.Write(buffer, 0, buffer.Length)
                        
                        | _ ->
                            response.StatusCode <- 404
                        
                        response.OutputStream.Close()
                    with 
                    | :? OperationCanceledException -> ()
                    | :? ObjectDisposedException -> ()
                    | ex -> 
                         // Console.WriteLine($"[Infra] Error handling request: {ex.Message}")
                         ()
            , cancellationToken)
        | None -> Task.CompletedTask