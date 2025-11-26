namespace Tars.Tests

open System
open System.Net
open System.Net.Sockets
open System.Threading.Tasks
open System.Text
open Xunit
open Tars.Connectors
open Tars.Security

type OpenWebUiTests(output: Xunit.Abstractions.ITestOutputHelper) =
    
    let getFreePort () =
        let l = new TcpListener(IPAddress.Loopback, 0)
        l.Start()
        let port = (l.LocalEndpoint :?> IPEndPoint).Port
        l.Stop()
        port

    [<Fact>]
    member this.``Can list models from Open WebUI``() =
        task {
            let port = getFreePort()
            let prefix = $"http://localhost:{port}/"
            use listener = new HttpListener()
            listener.Prefixes.Add(prefix)
            listener.Start()
            
            // Start a background loop to handle requests
            let serverLoop = task {
                try
                    while listener.IsListening do
                        let! context = listener.GetContextAsync()
                        let req = context.Request
                        let resp = context.Response
                        
                        let writeJson (json: string) =
                            let bytes = Encoding.UTF8.GetBytes(json)
                            resp.ContentType <- "application/json"
                            resp.ContentLength64 <- int64 bytes.Length
                            resp.OutputStream.Write(bytes, 0, bytes.Length)
                            resp.OutputStream.Close()

                        if req.Url.AbsolutePath.EndsWith("/api/v1/auths/signin") && req.HttpMethod = "POST" then
                             let json = """{"token": "fake_token"}"""
                             resp.StatusCode <- 200
                             writeJson json
                        elif req.Url.AbsolutePath.EndsWith("/api/models") && req.HttpMethod = "GET" then
                             let auth = req.Headers.["Authorization"]
                             if auth = "Bearer fake_token" then
                                 let json = """{
                                    "object": "list",
                                    "data": [
                                        { "id": "arena-model", "name": "Arena Model", "object": "model", "created": 123, "owned_by": "me" }
                                    ]
                                 }"""
                                 resp.StatusCode <- 200
                                 writeJson json
                             else
                                 resp.StatusCode <- 401
                                 let bytes = Encoding.UTF8.GetBytes("Unauthorized")
                                 resp.OutputStream.Write(bytes, 0, bytes.Length)
                                 resp.OutputStream.Close()
                        else
                            resp.StatusCode <- 404
                            resp.Close()
                with
                | :? ObjectDisposedException -> () // Listener stopped
                | :? HttpListenerException -> () // Listener closed
                | ex -> printfn "Server error: %s" ex.Message
            }

            try
                // Register secrets
                CredentialVault.registerSecret "OPENWEBUI_EMAIL" "test@example.com"
                CredentialVault.registerSecret "OPENWEBUI_PASSWORD" "password"
                
                // Run test
                let! result = OpenWebUi.listModels prefix
                
                match result with
                | Ok models ->
                    output.WriteLine("Models found:")
                    for m in models do
                        output.WriteLine($"Model: {m.name} (ID: {m.id})")

                    Assert.NotEmpty(models)
                    Assert.Contains(models, fun m -> m.id = "arena-model")
                    output.WriteLine("Confirmed 'arena-model' is available.")
                | Error e ->
                    output.WriteLine($"Error listing models: {e}")
                    Assert.Fail($"Failed to list models: {e}")
            finally
                listener.Stop() 
                try serverLoop.Wait() with _ -> ()
        }
