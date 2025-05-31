Here is the complete, functional code for `index.js`:

```javascript
// index.js
import React from 'react';
import ReactDOM from 'react-dom';
import Calculator from './components/Calculator';

const App = () => {
  return (
    <div>
      <h1>Simple Calculator</h1>
      <Calculator />
    </div>
  );
};

ReactDOM.render(<App />, document.getElementById('root'));
```

This code sets up a basic React application with a single component, `Calculator`, which is imported from the `components` directory. The `App` component renders the `Calculator` component and displays it on the page.

Here's a breakdown of the code:

1. We import `React` and `ReactDOM` from the `react` package.
2. We define the `App` component as a functional component that returns a JSX element.
3. The JSX element contains an `<h1>` heading with the text "Simple Calculator" and an instance of the `Calculator` component, which is imported from the `components` directory.
4. We use `ReactDOM.render()` to render the `App` component to the DOM, targeting the element with the ID "root".

Note that this code assumes you have a `public/index.html` file set up as your entry point, and that you've created a `components/Calculator.js` file containing the calculator component implementation.