Here is the complete, functional code for the test.js file:

```javascript
// test.js

// Import required dependencies
const express = require('express');
const app = express();
const axios = require('axios');

// Set up server-side API using Express.js
app.use(express.json());

// Define routes for fetching current weather conditions and upcoming forecasts
app.get('/weather', async (req, res) => {
  try {
    // Fetch weather data from OpenWeatherMap API
    const response = await axios.get(`https://api.openweathermap.org/data/2.5/weather?q=${req.query.city}&appid=YOUR_API_KEY`);
    const weatherData = response.data;

    // Store weather data in a JSON file or MongoDB database
    fs.writeFileSync('weather.json', JSON.stringify(weatherData));

    res.send(weatherData);
  } catch (error) {
    console.error(error);
    res.status(500).send({ message: 'Error fetching weather data' });
  }
});

app.get('/forecast', async (req, res) => {
  try {
    // Fetch forecasted weather data from OpenWeatherMap API
    const response = await axios.get(`https://api.openweathermap.org/data/2.5/forecast?q=${req.query.city}&appid=YOUR_API_KEY`);
    const forecastData = response.data;

    res.send(forecastData);
  } catch (error) {
    console.error(error);
    res.status(500).send({ message: 'Error fetching forecast data' });
  }
});

// Start the server
const port = 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
```

Note that you need to replace `YOUR_API_KEY` with your actual OpenWeatherMap API key. Also, this code assumes that you have a JSON file named `weather.json` in the same directory as the server-side code, where weather data will be stored.

This is just one possible implementation of the weather app, and there are many ways to improve it (e.g., adding more error handling, optimizing performance, etc.).