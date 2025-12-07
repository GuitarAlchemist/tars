namespace Tars.Sandbox

open System
open System.Threading.Tasks
open Docker.DotNet
open Docker.DotNet.Models

module DockerClient =
    let createClient () =
        let uri =
            if
                System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                    System.Runtime.InteropServices.OSPlatform.Windows
                )
            then
                Uri("npipe://./pipe/docker_engine")
            else
                Uri("unix:///var/run/docker.sock")

        (new DockerClientConfiguration(uri)).CreateClient()

    let runContainer (client: DockerClient) (image: string) (cmd: string list) =
        task {
            try
                // 1. Create Container
                let config = CreateContainerParameters()
                config.Image <- image
                config.Cmd <- ResizeArray(cmd)
                config.Tty <- false
                config.AttachStdout <- true
                config.AttachStderr <- true

                let! response = client.Containers.CreateContainerAsync(config)
                let id = response.ID

                // 2. Start Container
                let! started = client.Containers.StartContainerAsync(id, ContainerStartParameters())

                if not started then
                    return Error "Failed to start container"
                else
                    // 3. Wait for Container
                    let! waitResponse = client.Containers.WaitContainerAsync(id)
                    
                    // 4. Get Logs
                    let logParams = ContainerLogsParameters()
                    logParams.ShowStdout <- true
                    logParams.ShowStderr <- true
                    
                    // MultiplexedStream
                    let! stream = client.Containers.GetContainerLogsAsync(id, true, logParams)
                    let! (stdout, stderr) = stream.ReadOutputToEndAsync(System.Threading.CancellationToken.None)

                    return Ok (stdout, stderr, waitResponse.StatusCode)

            with ex ->
                return Error $"Docker Error: %s{ex.Message}"
        }
