namespace TarsEngineFSharp

open FSharp.Data
open System
open System.Net.Http

module WeatherService =
    type WeatherData = {
        Location: string
        Temperature: float
        Condition: string
        Provider: string
    }

    type WeatherProvider =
        | OpenMeteo
        | WeatherApi
        | OpenWeatherMap
        | IPBased

    // Helper to get location from IP
    let private getLocationFromIp() = async {
        try
            let! response = 
                Http.AsyncRequestString("http://ip-api.com/json/")
            let location = JsonValue.Parse(response)
            return Some(
                location.GetProperty("lat").AsFloat(),
                location.GetProperty("lon").AsFloat(),
                location.GetProperty("city").AsString()
            )
        with _ ->
            return None
    }

    // Get coordinates from city name using different providers
    let private getCoordinatesFromCity (city: string) = async {
        let geocodingProviders = [
            async {
                // OpenMeteo Geocoding
                try
                    let encodedCity = Uri.EscapeDataString(city)
                    let! response = Http.AsyncRequestString $"https://geocoding-api.open-meteo.com/v1/search?name=%s{encodedCity}&count=1"
                    let result = JsonValue.Parse(response)
                    let results = result.GetProperty("results").AsArray()
                    if results.Length > 0 then
                        return Some(
                            results.[0].GetProperty("latitude").AsFloat(),
                            results.[0].GetProperty("longitude").AsFloat(),
                            results.[0].GetProperty("name").AsString()
                        )
                    else return None
                with _ -> return None
            }
            
            async {
                // Nominatim (OpenStreetMap) as backup
                try
                    let encodedCity = Uri.EscapeDataString(city)
                    let! response = Http.AsyncRequestString(
                        $"https://nominatim.openstreetmap.org/search?q=%s{encodedCity}&format=json&limit=1",
                        headers = [ "User-Agent", "TarsEngine/1.0" ]
                    )
                    let searchResult = JsonValue.Parse(response)
                    if searchResult.AsArray().Length > 0 then
                        let result = searchResult.[0]
                        return Some(
                            result.GetProperty("lat").AsFloat(),
                            result.GetProperty("lon").AsFloat(),
                            result.GetProperty("display_name").AsString().Split(',').[0]
                        )
                    else return None
                with _ -> return None
            }
        ]

        // Try each provider until we get coordinates
        let! results = geocodingProviders |> Async.Parallel
        return results |> Array.tryPick id
    }

    // Weather providers
    let private getOpenMeteoWeather (latitude: float) (longitude: float) = async {
        try
            use client = new HttpClient()
            let url = $"https://api.open-meteo.com/v1/forecast?latitude=%f{latitude}&longitude=%f{longitude}&current=temperature_2m,weather_code&timezone=auto"
            let! response = client.GetStringAsync(url) |> Async.AwaitTask
            let weather = JsonValue.Parse(response)
            let current = weather.GetProperty("current")
            return Some(
                current.GetProperty("temperature_2m").AsFloat(),
                current.GetProperty("weather_code").AsInteger()
            )
        with _ -> return None
    }

    let private getWeatherApiWeather (latitude: float) (longitude: float) = async {
        // Note: Requires API key in practice
        try
            use client = new HttpClient()
            let url = $"https://api.weatherapi.com/v1/current.json?q=%f{latitude},%f{longitude}&key=YOUR_API_KEY"
            let! response = client.GetStringAsync(url) |> Async.AwaitTask
            let weather = JsonValue.Parse(response)
            let current = weather.GetProperty("current")
            return Some(
                current.GetProperty("temp_c").AsFloat(),
                current.GetProperty("condition").GetProperty("code").AsInteger()
            )
        with _ -> return None
    }

    // Convert WMO weather code to description
    let private weatherCodeToCondition (code: int) =
        match code with
        | 0 -> "Clear sky"
        | 1 | 2 | 3 -> "Partly cloudy"
        | 45 | 48 -> "Foggy"
        | 51 | 53 | 55 -> "Drizzle"
        | 61 | 63 | 65 -> "Rain"
        | 71 | 73 | 75 -> "Snow"
        | 77 -> "Snow grains"
        | 80 | 81 | 82 -> "Rain showers"
        | 85 | 86 -> "Snow showers"
        | 95 -> "Thunderstorm"
        | 96 | 99 -> "Thunderstorm with hail"
        | _ -> "Unknown"

    // Main weather fetching function with provider selection
    let fetchWeatherData (location: string option) (provider: WeatherProvider option) = async {
        try
            let selectedProvider = defaultArg provider OpenMeteo
            
            let! locationData = 
                match location with
                | None -> getLocationFromIp()
                | Some loc when String.IsNullOrEmpty(loc) -> getLocationFromIp()
                | Some loc -> getCoordinatesFromCity loc

            match locationData with
            | Some(lat, lon, cityName) ->
                let! weatherData = 
                    match selectedProvider with
                    | OpenMeteo -> 
                        async {
                            let! result = getOpenMeteoWeather lat lon
                            match result with
                            | Some(temp, code) -> 
                                return Some({
                                    Location = cityName
                                    Temperature = temp
                                    Condition = weatherCodeToCondition code
                                    Provider = "OpenMeteo"
                                })
                            | None -> return None
                        }
                    | WeatherApi ->
                        async {
                            let! result = getWeatherApiWeather lat lon
                            match result with
                            | Some(temp, code) ->
                                return Some({
                                    Location = cityName
                                    Temperature = temp
                                    Condition = weatherCodeToCondition code
                                    Provider = "WeatherAPI"
                                })
                            | None -> return None
                        }
                    | OpenWeatherMap ->
                        // Implementation similar to above
                        async { return None }
                    | IPBased ->
                        async {
                            let! result = getOpenMeteoWeather lat lon
                            match result with
                            | Some(temp, code) ->
                                return Some({
                                    Location = cityName
                                    Temperature = temp
                                    Condition = weatherCodeToCondition code
                                    Provider = "IP-Based"
                                })
                            | None -> return None
                        }

                match weatherData with
                | Some data -> return data
                | None -> 
                    // Fallback to OpenMeteo if selected provider fails
                    let! fallbackResult = getOpenMeteoWeather lat lon
                    match fallbackResult with
                    | Some(temp, code) ->
                        return {
                            Location = cityName
                            Temperature = temp
                            Condition = weatherCodeToCondition code
                            Provider = "OpenMeteo (Fallback)"
                        }
                    | None ->
                        return {
                            Location = cityName
                            Temperature = 20.5
                            Condition = "Unknown"
                            Provider = "Fallback"
                        }
            | None ->
                return {
                    Location = Option.defaultValue "Unknown" location
                    Temperature = 20.5
                    Condition = "Unknown"
                    Provider = "None"
                }
        with ex ->
            return {
                Location = Option.defaultValue "Unknown" location
                Temperature = 20.5
                Condition = "Error fetching weather"
                Provider = "Error"
            }
    }

    // Helper functions for common use cases
    let getWeatherByCity (city: string) = 
        fetchWeatherData (Some city) None

    let getWeatherByProvider (city: string) (provider: WeatherProvider) =
        fetchWeatherData (Some city) (Some provider)

    let getLocalWeather() =
        fetchWeatherData None (Some IPBased)
