module APIDataFetcher

open System
open System.Net.Http
open Newtonsoft.Json.Linq

type WeatherData = {
    Temperature: float
    Condition: string
    Location: string
    Timestamp: DateTime
}

type NewsItem = {
    Title: string
    Url: string
    Source: string
    PublishedAt: DateTime
}

let private httpClient = new HttpClient()

let fetchWeatherData location =
    async {
        let apiKey = Environment.GetEnvironmentVariable("WEATHER_API_KEY")
        let url = sprintf "https://api.openweathermap.org/data/2.5/weather?q=%s&appid=%s&units=metric"
                    (Uri.EscapeDataString(location)) apiKey
        
        let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
        let data = JObject.Parse(response)
        
        return {
            Temperature = float(data.["main"].["temp"].ToString())
            Condition = data.["weather"].[0].["main"].ToString()
            Location = location
            Timestamp = DateTime.UtcNow
        }
    }

let fetchLatestNews () =
    async {
        let apiKey = Environment.GetEnvironmentVariable("NEWS_API_KEY")
        let url = sprintf "https://newsapi.org/v2/top-headlines?country=us&apiKey=%s" apiKey
        
        let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
        let data = JObject.Parse(response)
        
        return data.["articles"]
               |> Seq.map (fun article -> 
                   { Title = article.["title"].ToString()
                     Url = article.["url"].ToString()
                     Source = article.["source"].["name"].ToString()
                     PublishedAt = DateTime.Parse(article.["publishedAt"].ToString()) })
               |> Seq.toList
    }