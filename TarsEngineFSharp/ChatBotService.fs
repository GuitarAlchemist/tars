module TarsEngineFSharp.ChatBotService

open WeatherService
open TarsEngineFSharp.LlmService

type ChatResponse = {
    Text: string
    Source: string
}

type ChatBot() =
    member _.ProcessMessage(message: string) = async {
        // Register available functions
        let functions = [
            {
                Name = "get_weather"
                Description = "Get the current weather for a location"
                Parameters = [
                    {
                        Name = "location"
                        Description = "The city or location to get weather for"
                        Required = true
                        Type = "string"
                    }
                ]
                Handler = fun args -> async {
                    let location = args.["location"].ToString() |> Some
                    let! weather = WeatherService.fetchWeatherData location None
                    return sprintf "In %s, the temperature is %.1f°C and the condition is %s"
                            weather.Location weather.Temperature weather.Condition
                }
            }
        ]
        
        // Let LLM decide which function to call based on the message
        let! result = LlmService.processWithFunctions message functions
        
        return {
            Text = result
            Source = "LLM Service"
        }
    }

let createDefaultChatBot() = ChatBot()
