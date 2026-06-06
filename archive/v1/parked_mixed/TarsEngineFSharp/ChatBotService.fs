namespace TarsEngineFSharp

open WeatherService

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
                Handler = fun args ->
                    async {
                        let location = args.["location"].ToString()
                        let! weather = WeatherService.fetchWeatherData location
                        return sprintf "In %s, the temperature is %.1f°C and the condition is %s"
                                weather.Location weather.Temperature weather.Condition
                    }
            }
        ]
        
        // Let LLM decide which function to call based on the message
        let! result = LlmService.processWithFunctions message functions
        return { Text = result; Source = "ChatBot" }
    }

type IChatBotService =
    abstract member GetResponse : string -> System.Threading.Tasks.Task<ChatResponse>

type ChatBotService() =
    let bot = ChatBot()
    
    interface IChatBotService with
        member this.GetResponse(message: string) =
            bot.ProcessMessage(message)
            |> Async.StartAsTask

let createChatBotService() = ChatBotService() :> IChatBotService