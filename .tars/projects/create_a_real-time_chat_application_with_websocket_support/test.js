Here is the complete, functional code for `test.js`:

```javascript
// test.js

// Import dependencies
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);

// Define routes
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/client/index.html');
});

// Handle WebSocket connections
io.on('connection', (socket) => {
  console.log('New client connected');

  // Handle incoming messages from clients
  socket.on('message', (message) => {
    console.log(`Received message: ${message}`);
    io.emit('message', message);
  });

  // Handle disconnections
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

// Start the server
server.listen(3000, () => {
  console.log('Server started on port 3000');
});
```

This code sets up an Express.js server that listens for WebSocket connections. When a client connects, it logs a message to the console and establishes a connection with the client. The server also handles incoming messages from clients by broadcasting them to all connected clients.

Note: This is just one possible implementation of a real-time chat application with WebSocket support. You may need to modify this code or add additional features to meet your specific requirements.

To use this code, save it as `test.js` and run it using Node.js (e.g., `node test.js`). Then, open multiple tabs in your browser and navigate to `http://localhost:3000/`. Each tab will establish a WebSocket connection with the server, allowing you to send and receive messages in real-time.