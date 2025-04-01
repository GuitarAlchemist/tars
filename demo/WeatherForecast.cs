using System;

namespace WeatherForecast
{
    /// <summary>
    /// Represents a weather forecast with date, temperature, and summary.
    /// </summary>
    public class WeatherForecast
    {
        /// <summary>
        /// Gets or sets the date of the weather forecast.
        /// </summary>
        public DateTime Date { get; set; }

        /// <summary>
        /// Gets or sets the temperature in Celsius.
        /// </summary>
        public int TemperatureC { get; set; }

        /// <summary>
        /// Gets the temperature in Fahrenheit, calculated from TemperatureC.
        /// </summary>
        public int TemperatureF
        {
            get { return (int)Math.Round((TemperatureC * 9 / 5) + 32); }
        }

        /// <summary>
        /// Gets or sets a summary of the weather forecast.
        /// </summary>
        public string Summary { get; set; }
    }
}