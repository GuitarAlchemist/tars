Here is the complete `index.js` file content:

```javascript
// Import dependencies
const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);

// Set up server configuration
const port = 3000;

app.use(express.static('public'));

// Define WebSocket routes
io.on('connection', (socket) => {
    console.log('A new client connected');

    // Handle incoming message from the client
    socket.on('message', (message) => {
        console.log(`Received message: ${message}`);
        io.emit('message', message);
    });

    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

// Start the server
http.listen(port, () => {
    console.log(`Server started on port ${port}`);
});
```

This code sets up an Express.js server that uses Socket.IO to establish a WebSocket connection with clients. When a client connects, it logs a message and sets up event listeners for incoming messages and disconnections.

Note: This is just the `index.js` file content, which is part of the overall project structure outlined in the analysis. You will need to create additional files (e.g., `routes.js`, `models.js`, `config.js`) and directories (e.g., `server`, `client`, `public`) as described in the analysis.