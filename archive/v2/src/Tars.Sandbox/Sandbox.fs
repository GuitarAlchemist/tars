namespace Tars.Sandbox

open System
open Docker.DotNet
open Docker.DotNet.Models

module DockerClient =
    let createClient () =
        if
            System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows
            )
        then
            new DockerClientConfiguration(Uri("npipe://./pipe/docker_engine"))
        else
            new DockerClientConfiguration(Uri("unix:///var/run/docker.sock"))
        |> fun config -> config.CreateClient()

    let runContainer (client: DockerClient) (image: string) (cmd: string list) =
        task {
            let mutable containerId = ""

            // Helper to clean up securely
            let cleanup () =
                task {
                    if not (String.IsNullOrEmpty containerId) then
                        try
                            do!
                                client.Containers.RemoveContainerAsync(
                                    containerId,
                                    ContainerRemoveParameters(Force = true)
                                )
                        with _ ->
                            ()
                }

            try
                try
                    // 1. Create Container
                    let config = CreateContainerParameters()
                    config.Image <- image
                    config.Cmd <- ResizeArray(cmd)
                    config.Tty <- false
                    config.AttachStdout <- true
                    config.AttachStderr <- true
                    config.HostConfig <- HostConfig(NetworkMode = "none")

                    let! response = client.Containers.CreateContainerAsync(config)
                    containerId <- response.ID

                    // 2. Start Container
                    let! started = client.Containers.StartContainerAsync(containerId, ContainerStartParameters())

                    if not started then
                        do! cleanup ()
                        return Error "Failed to start container"
                    else
                        // 3. Wait for Container
                        let! waitResponse = client.Containers.WaitContainerAsync(containerId)

                        // 4. Get Logs
                        let logParams = ContainerLogsParameters()
                        logParams.ShowStdout <- true
                        logParams.ShowStderr <- true

                        // MultiplexedStream
                        let! stream = client.Containers.GetContainerLogsAsync(containerId, false, logParams)
                        let! (stdout, stderr) = stream.ReadOutputToEndAsync(System.Threading.CancellationToken.None)

                        do! cleanup ()
                        return Ok(stdout, stderr, waitResponse.StatusCode)

                with ex ->
                    do! cleanup ()
                    return Error $"Docker Error: %s{ex.Message}"
            with ex ->
                // Fallback for any catastrophic error in cleanup logic itself
                return Error $"Critical Docker Error: %s{ex.Message}"
        }
