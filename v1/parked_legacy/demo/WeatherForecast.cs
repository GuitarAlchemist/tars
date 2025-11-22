using System;

namespace WeatherForecasts
{
    /// <summary>
    /// Represents a weather forecast for a given date.
    /// </summary>
    public class WeatherForecast
    {
        /// <summary>
        /// Gets or sets the date of the weather forecast.
        /// </summary>
        /// <value>A DateTime object representing the date.</value>
        public DateTime Date { get; set; }

        /// <summary>
        /// Gets or sets the temperature in Celsius.
        /// </summary>
        /// <value>An integer value representing the temperature in Celsius.</value>
        public int TemperatureC { get; set; }

        /// <summary>
        /// Gets the temperature in Fahrenheit, calculated from the TemperatureC property.
        /// </summary>
        /// <value>A string value representing the temperature in Fahrenheit.</value>
        public string TemperatureF
        {
            get
            {
                return $"{(TemperatureC * 9 / 5) + 32}°F";
            }
        }

        /// <summary>
        /// Gets or sets a summary of the weather forecast.
        /// </summary>
        /// <value>A string value representing the summary.</value>
        public string Summary { get; set; }
    }
}