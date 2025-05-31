module TarsDslSpeechExtensions

open System
open System.Collections.Generic
open System.IO
open System.Threading.Tasks
open TarsTtsAgent

// Represents the speech config parsed from the DSL
module SpeechConfig =
    type Config = {
        Enabled: bool
        AutoSpeakOnReflection: bool
        DefaultVoice: string
        LogTranscripts: bool
        Language: string
        PreloadVoices: bool
        MaxConcurrency: int
        VoiceModels: Map<string, string>
    }

    let defaultConfig = {
        Enabled = true
        AutoSpeakOnReflection = true
        DefaultVoice = "tts_models/en/ljspeech/tacotron2-DDC"
        LogTranscripts = true
        Language = "en"
        PreloadVoices = true
        MaxConcurrency = 1
        VoiceModels = Map.empty
            .Add("en", "tts_models/en/ljspeech/tacotron2-DDC")
            .Add("fr", "tts_models/fr/mai/tacotron2-DDC")
            .Add("es", "tts_models/es/mai/tacotron2-DDC")
            .Add("de", "tts_models/de/thorsten/tacotron2-DDC")
            .Add("it", "tts_models/it/mai_female/glow-tts")
            .Add("nl", "tts_models/nl/mai/tacotron2-DDC")
            .Add("ru", "tts_models/ru/multi-dataset/vits")
            .Add("clone", "tts_models/multilingual/multi-dataset/your_tts")
    }

    let mutable activeConfig = defaultConfig
    let mutable loadedVoices = HashSet<string>()

    let apply (cfg: Config) =
        activeConfig <- cfg
        
        // Preload voices if enabled
        if cfg.PreloadVoices then
            let modelsToPreload = 
                cfg.VoiceModels
                |> Map.values
                |> Seq.distinct
                |> Seq.toList
            
            loadedVoices.Clear()
            for model in modelsToPreload do
                loadedVoices.Add(model)
            
            preloadModels modelsToPreload |> Async.AwaitTask |> Async.RunSynchronously

    let getModelForLanguage (lang: string) =
        match Map.tryFind lang activeConfig.VoiceModels with
        | Some model -> model
        | None -> activeConfig.DefaultVoice

// Speech execution functions
let executeSpeak (text: string, ?voiceOverride: string, ?langOverride: string, ?speakerWav: string, ?agentId: string) =
    if SpeechConfig.activeConfig.Enabled then
        let voice = 
            match voiceOverride with
            | Some v -> v
            | None -> 
                match langOverride with
                | Some lang -> SpeechConfig.getModelForLanguage lang
                | None -> SpeechConfig.activeConfig.DefaultVoice
        
        let language = 
            match langOverride with
            | Some lang -> lang
            | None -> SpeechConfig.activeConfig.Language
        
        speakAndPlay(text, voice, language, ?speakerWav=speakerWav, ?agentId=agentId)
        |> Async.AwaitTask
        |> Async.Start

// Execute multiple speak commands
let executeSpeakMulti (items: SpeakItem list) =
    if SpeechConfig.activeConfig.Enabled then
        speakMulti(items, SpeechConfig.activeConfig.MaxConcurrency)
        |> Async.AwaitTask
        |> Async.Start

// Parse speech_module block from DSL
let parseSpeechModule (block: obj) =
    let dict = block :?> Dictionary<string, obj>
    
    let tryGetValue<'T> (key: string) (defaultValue: 'T) =
        if dict.ContainsKey(key) then
            try
                Convert.ChangeType(dict.[key], typeof<'T>) :?> 'T
            with _ ->
                defaultValue
        else
            defaultValue
    
    let config = {
        SpeechConfig.Config.Enabled = tryGetValue "enabled" SpeechConfig.defaultConfig.Enabled
        AutoSpeakOnReflection = tryGetValue "auto_speak_on_reflection" SpeechConfig.defaultConfig.AutoSpeakOnReflection
        DefaultVoice = tryGetValue "default_voice" SpeechConfig.defaultConfig.DefaultVoice
        LogTranscripts = tryGetValue "log_transcripts" SpeechConfig.defaultConfig.LogTranscripts
        Language = tryGetValue "language" SpeechConfig.defaultConfig.Language
        PreloadVoices = tryGetValue "preload_voices" SpeechConfig.defaultConfig.PreloadVoices
        MaxConcurrency = tryGetValue "max_concurrency" SpeechConfig.defaultConfig.MaxConcurrency
        VoiceModels = SpeechConfig.defaultConfig.VoiceModels
    }
    
    // Parse voice models if present
    if dict.ContainsKey("voice_models") then
        let voiceModels = dict.["voice_models"] :?> Dictionary<string, obj>
        let mutable updatedVoiceModels = SpeechConfig.defaultConfig.VoiceModels
        
        for KeyValue(lang, model) in voiceModels do
            updatedVoiceModels <- updatedVoiceModels.Add(lang, model.ToString())
        
        SpeechConfig.apply { config with VoiceModels = updatedVoiceModels }
    else
        SpeechConfig.apply config

// Parse speak_extended block from DSL
let parseSpeakExtended (block: obj) =
    let dict = block :?> Dictionary<string, obj>
    
    let text = 
        if dict.ContainsKey("text") then dict.["text"].ToString()
        else ""
    
    let voice = 
        if dict.ContainsKey("voice") then Some(dict.["voice"].ToString())
        else None
    
    let language = 
        if dict.ContainsKey("language") then Some(dict.["language"].ToString())
        else None
    
    let speakerWav = 
        if dict.ContainsKey("speaker_wav") then Some(dict.["speaker_wav"].ToString())
        else None
    
    let agentId = 
        if dict.ContainsKey("agentId") then Some(dict.["agentId"].ToString())
        else None
    
    executeSpeak(text, ?voiceOverride=voice, ?langOverride=language, ?speakerWav=speakerWav, ?agentId=agentId)

// Parse speak_multi block from DSL
let parseSpeakMulti (block: obj) =
    let items = block :?> System.Collections.IEnumerable
    let speakItems = ResizeArray<SpeakItem>()
    
    for item in items do
        let dict = item :?> Dictionary<string, obj>
        
        let text = 
            if dict.ContainsKey("text") then dict.["text"].ToString()
            else ""
        
        let voice = 
            if dict.ContainsKey("voice") then Some(dict.["voice"].ToString())
            else None
        
        let language = 
            if dict.ContainsKey("language") then Some(dict.["language"].ToString())
            else None
        
        let speakerWav = 
            if dict.ContainsKey("speaker_wav") then Some(dict.["speaker_wav"].ToString())
            else None
        
        let agentId = 
            if dict.ContainsKey("agentId") then Some(dict.["agentId"].ToString())
            else None
        
        speakItems.Add({
            Text = text
            Voice = voice
            Language = language
            SpeakerWav = speakerWav
            AgentId = agentId
        })
    
    executeSpeakMulti(speakItems |> Seq.toList)

// Initialize the speech system
let initialize() =
    // Ensure Python and TTS are installed
    installPythonIfMissing()
    installTts()
    
    // Start the TTS server
    let server = startServer()
    
    // Apply default config
    SpeechConfig.apply SpeechConfig.defaultConfig
    
    // Create spoken_trace.tars if it doesn't exist
    if not (File.Exists("spoken_trace.tars")) then
        File.WriteAllText("spoken_trace.tars", "# TARS Speech Transcript\n\n")

// Cleanup
let shutdown() =
    stopServer()
