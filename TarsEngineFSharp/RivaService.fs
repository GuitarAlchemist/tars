namespace TarsEngineFSharp

open System
open System.Threading.Tasks

module RivaService =
    type Intent = {
        Name: string
        Confidence: float
    }

    type Entity = {
        Type: string
        Value: string
        Position: int * int
    }

    type ProcessingResult = {
        Intent: Intent option
        Entities: Entity list
        OriginalText: string
    }

    type AudioResponse = {
        AudioData: byte array
        SampleRate: int
        Channels: int
    }

    type RivaConfig = {
        Language: string
        VoiceId: string
    }

    type RivaClient(config: RivaConfig) =
        let defaultIntent = {
            Name = "query"
            Confidence = 0.85
        }

        let mockEntities = [
            { Type = "DATE"; Value = "today"; Position = (0, 5) }
            { Type = "LOCATION"; Value = "New York"; Position = (10, 18) }
        ]

        member _.ProcessQuery(text: string) = async {
            // Simulate processing delay
            do! Async.Sleep 100
            
            return {
                Intent = Some defaultIntent
                Entities = mockEntities
                OriginalText = text
            }
        }

        member _.GenerateResponse(text: string) = async {
            // Simulate audio generation delay
            do! Async.Sleep 200
            
            return {
                AudioData = Array.zeroCreate 1024 // Mock audio data
                SampleRate = 16000
                Channels = 1
            }
        }

        interface IDisposable with
            member _.Dispose() = ()

    // Initialize Riva client with default configuration
    let createDefaultClient() =
        let config = {
            Language = "en-US"
            VoiceId = "female-1"
        }
        new RivaClient(config)