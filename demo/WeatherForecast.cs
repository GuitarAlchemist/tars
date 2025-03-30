using System;

namespace WeatherApi
{
    /// <summary>
    /// Represents a weather forecast with date, temperature, and summary.
    /// </summary>
    public class WeatherForecast
    {
        /// <summary>
        /// Gets or sets the date of the weather forecast.
        /// </summary>
        /// <value>A DateTime representing the date of the weather forecast.</value>
        public DateTime Date { get; set; }

        /// <summary>
        /// Gets or sets the temperature in Celsius.
        /// </summary>
        /// <value>An integer representing the temperature in Celsius.</value>
        public int TemperatureC { get; set; }

        /// <summary>
        /// Gets the temperature in Fahrenheit, calculated from TemperatureC.
        /// </summary>
        /// <value>A string representing the temperature in Fahrenheit (e.g. "75°F").</value>
        public string TemperatureF => $"{TemperatureC * 9 / 5 + 32}°F";

        /// <summary>
        /// Gets or sets a brief summary of the weather forecast.
        /// </summary>
        /// <value>A string representing the weather summary.</value>
        public string Summary { get; set; }
    }
}