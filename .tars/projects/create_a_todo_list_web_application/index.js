Here is the complete, functional code for `index.js`:

```javascript
// Import dependencies
const express = require('express');
const app = express();
const mongoose = require('mongoose');
const passport = require('passport');
const session = require('express-session');

// Connect to MongoDB
mongoose.connect('mongodb://localhost/todo-list-app', { useNewUrlParser: true, useUnifiedTopology: true });

// Set up Express.js and Passport.js
app.use(express.urlencoded({ extended: true }));
app.use(session({
  secret: 'secret',
  resave: false,
  saveUninitialized: true,
  cookie: {}
}));
app.use(passport.initialize());
app.use(passport.session());

// Define API routes
const apiRoutes = require('./routes/api');
app.use('/api', apiRoutes);

// Serve static assets (CSS, images, etc.)
app.use(express.static('public'));

// Set up view engine (EJS)
app.set('view engine', 'ejs');

// Define main route for the application
app.get('/', (req, res) => {
  res.render('index');
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
```

This code sets up an Express.js server with Passport.js for user authentication. It also defines API routes using the `apiRoutes` file and serves static assets from the `public` directory. The main route is set to render the `index.ejs` template, which will display the todo list.

Note that this code assumes you have already created the necessary files and directories as outlined in the project structure. You will need to create these files and directories before running this code.