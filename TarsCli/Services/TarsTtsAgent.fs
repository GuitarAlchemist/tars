module TarsTtsAgent

open System
open System.IO
open System.Net.Http
open System.Text
open System.Text.Json
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open System.Collections.Generic
open System.Runtime.InteropServices

// OS detection
let isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
let isLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
let isMac = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)

// Python checks
let run (cmd: string) (args: string) =
    let psi = ProcessStartInfo(cmd, args)
    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    psi.UseShellExecute <- false
    psi.CreateNoWindow <- true
    use p = Process.Start(psi)
    p.WaitForExit()
    p.ExitCode = 0

let pythonExists () =
    try run "python" "--version"
    with _ -> false

let installPythonIfMissing () =
    if not (pythonExists()) then
        failwith "Python is required. Install manually or add auto-install logic."

let installTts () =
    run "python" "-m pip install TTS flask soundfile numpy langdetect"

// Server management
let mutable serverProcess: Process option = None

let startServer () =
    match serverProcess with
    | Some p when not p.HasExited -> p
    | _ ->
        let scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Python", "tts_server.py")
        let psi = ProcessStartInfo("python", scriptPath)
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        let proc = new Process()
        proc.StartInfo <- psi
        proc.Start() |> ignore
        
        // Wait for server to start
        System.Threading.Thread.Sleep(3000)
        
        serverProcess <- Some proc
        proc

let stopServer () =
    match serverProcess with
    | Some p when not p.HasExited -> 
        p.Kill()
        serverProcess <- None
    | _ -> ()

// TTS API calls
let speakToBytes (text: string, ?voice: string, ?language: string, ?speakerWav: string, ?agentId: string) =
    task {
        use client = new HttpClient()
        
        let requestObj = 
            match voice, language, speakerWav, agentId with
            | Some v, Some l, Some s, Some a -> {| text = text; voice = v; language = l; speaker_wav = s; agentId = a |}
            | Some v, Some l, Some s, None -> {| text = text; voice = v; language = l; speaker_wav = s |}
            | Some v, Some l, None, Some a -> {| text = text; voice = v; language = l; agentId = a |}
            | Some v, None, Some s, Some a -> {| text = text; voice = v; speaker_wav = s; agentId = a |}
            | None, Some l, Some s, Some a -> {| text = text; language = l; speaker_wav = s; agentId = a |}
            | Some v, Some l, None, None -> {| text = text; voice = v; language = l |}
            | Some v, None, Some s, None -> {| text = text; voice = v; speaker_wav = s |}
            | Some v, None, None, Some a -> {| text = text; voice = v; agentId = a |}
            | None, Some l, Some s, None -> {| text = text; language = l; speaker_wav = s |}
            | None, Some l, None, Some a -> {| text = text; language = l; agentId = a |}
            | None, None, Some s, Some a -> {| text = text; speaker_wav = s; agentId = a |}
            | Some v, None, None, None -> {| text = text; voice = v |}
            | None, Some l, None, None -> {| text = text; language = l |}
            | None, None, Some s, None -> {| text = text; speaker_wav = s |}
            | None, None, None, Some a -> {| text = text; agentId = a |}
            | None, None, None, None -> {| text = text |}
        
        let json = JsonSerializer.Serialize(requestObj)
        use content = new StringContent(json, Encoding.UTF8, "application/json")
        let! response = client.PostAsync("http://localhost:5002/speak", content)
        response.EnsureSuccessStatusCode() |> ignore
        let! bytes = response.Content.ReadAsByteArrayAsync()
        return bytes
    }

// Cross-platform audio playback
let playAudio (wavBytes: byte[]) =
    if isWindows then
        // Windows: Use NAudio if available, otherwise save to temp file and play with PowerShell
        try
            #if WINDOWS
            use stream = new MemoryStream(wavBytes)
            use reader = new NAudio.Wave.WaveFileReader(stream)
            use output = new NAudio.Wave.WaveOutEvent()
            output.Init(reader)
            output.Play()
            while output.PlaybackState = NAudio.Wave.PlaybackState.Playing do
                System.Threading.Thread.Sleep(100)
            #else
            let tempFile = Path.Combine(Path.GetTempPath(), $"tars_speech_{Guid.NewGuid()}.wav")
            File.WriteAllBytes(tempFile, wavBytes)
            let psi = ProcessStartInfo("powershell", $"-c (New-Object Media.SoundPlayer '{tempFile}').PlaySync()")
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true
            use p = Process.Start(psi)
            p.WaitForExit()
            File.Delete(tempFile)
            #endif
        with ex ->
            printfn "Error playing audio: %s" ex.Message
    elif isLinux then
        // Linux: Use aplay
        let tempFile = Path.Combine(Path.GetTempPath(), $"tars_speech_{Guid.NewGuid()}.wav")
        File.WriteAllBytes(tempFile, wavBytes)
        let psi = ProcessStartInfo("aplay", tempFile)
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true
        use p = Process.Start(psi)
        p.WaitForExit()
        File.Delete(tempFile)
    elif isMac then
        // macOS: Use afplay
        let tempFile = Path.Combine(Path.GetTempPath(), $"tars_speech_{Guid.NewGuid()}.wav")
        File.WriteAllBytes(tempFile, wavBytes)
        let psi = ProcessStartInfo("afplay", tempFile)
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true
        use p = Process.Start(psi)
        p.WaitForExit()
        File.Delete(tempFile)
    else
        printfn "Unsupported platform for audio playback"

// Speak and play
let speakAndPlay (text: string, ?voice: string, ?language: string, ?speakerWav: string, ?agentId: string) =
    task {
        let! bytes = speakToBytes(text, ?voice=voice, ?language=language, ?speakerWav=speakerWav, ?agentId=agentId)
        playAudio bytes
    }

// Speak multiple texts in sequence
let speakSequence (texts: seq<string>, ?voice: string, ?language: string, ?speakerWav: string, ?agentId: string) =
    task {
        for text in texts do
            let! _ = speakAndPlay(text, ?voice=voice, ?language=language, ?speakerWav=speakerWav, ?agentId=agentId)
            ()
    }

// Speak multiple texts with different settings
type SpeakItem = {
    Text: string
    Voice: string option
    Language: string option
    SpeakerWav: string option
    AgentId: string option
}

let speakMulti (items: seq<SpeakItem>, maxConcurrency: int) =
    task {
        use semaphore = new SemaphoreSlim(maxConcurrency)
        let tasks = ResizeArray<Task>()
        
        for item in items do
            do! semaphore.WaitAsync()
            
            let task = 
                Task.Run(fun () ->
                    task {
                        try
                            do! speakAndPlay(
                                item.Text, 
                                ?voice=item.Voice, 
                                ?language=item.Language, 
                                ?speakerWav=item.SpeakerWav, 
                                ?agentId=item.AgentId)
                        finally
                            semaphore.Release() |> ignore
                    })
            
            tasks.Add(task)
        
        do! Task.WhenAll(tasks)
    }

// Preload models
let preloadModels (models: seq<string>) =
    task {
        use client = new HttpClient()
        let json = JsonSerializer.Serialize({| models = models |})
        use content = new StringContent(json, Encoding.UTF8, "application/json")
        let! response = client.PostAsync("http://localhost:5002/preload", content)
        response.EnsureSuccessStatusCode() |> ignore
    }

// Get server status
let getServerStatus () =
    task {
        use client = new HttpClient()
        let! response = client.GetAsync("http://localhost:5002/status")
        response.EnsureSuccessStatusCode() |> ignore
        let! content = response.Content.ReadAsStringAsync()
        return JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(content)
    }
