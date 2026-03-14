Here is the complete, functional code for `test.js`:

```javascript
// test.js

// Import dependencies
const express = require('express');
const app = express();
const mongoose = require('mongoose');
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;
const bcrypt = require('bcryptjs');

// Connect to MongoDB
mongoose.connect('mongodb://localhost/todo-list-app', { useNewUrlParser: true, useUnifiedTopology: true });

// Define Todo model
const todoSchema = new mongoose.Schema({
  title: String,
  description: String,
  dueDate: Date,
  priority: Number
});
const Todo = mongoose.model('Todo', todoSchema);

// Define User model
const userSchema = new mongoose.Schema({
  username: String,
  password: String
});
const User = mongoose.model('User', userSchema);

// Set up Passport.js
passport.use(new LocalStrategy({
  usernameField: 'username',
  passwordField: 'password'
}, (username, password, done) => {
  User.findOne({ username: username }, (err, user) => {
    if (err) { return done(err); }
    if (!user) { return done(null, false); }
    bcrypt.compare(password, user.password, (err, result) => {
      if (err) { return done(err); }
      if (!result) { return done(null, false); }
      return done(null, user);
    });
  });
}));

// Set up API routes
app.use('/api', require('./routes/api'));

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
```

This code sets up a basic Express.js server with Passport.js for user authentication and Mongoose for interacting with the MongoDB database. It also defines two models: `Todo` and `User`. The API routes are set up to handle CRUD operations for todo items.

Note that this is just a starting point, and you will need to add more functionality (such as error handling and logging) to make the application robust and scalable.