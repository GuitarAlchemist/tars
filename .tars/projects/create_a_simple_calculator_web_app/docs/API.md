# Code Documentation

## File Structure

### index.html
Main HTML structure with calculator layout and buttons.

### style.css
CSS styling including:
- Responsive grid layout
- Button styling and animations
- Color scheme and typography
- Mobile-first design

### index.js
JavaScript functionality including:

#### Main Functions
- \ppendToDisplay(value)\: Add value to calculator display
- \clearDisplay()\: Clear the current display
- \calculate()\: Evaluate the mathematical expression
- \handleKeyboard(event)\: Process keyboard input

#### Event Listeners
- Button click handlers
- Keyboard event handlers
- Window load handlers

## Code Examples

### Adding a New Operation
\\\javascript
// Add to the calculate function
case 'sqrt':
    result = Math.sqrt(parseFloat(display.value));
    break;
\\\

### Customizing Styles
\\\css
/* Change button colors */
.btn {
    background-color: #your-color;
    color: #text-color;
}
\\\

## Extension Points
- Add scientific calculator functions
- Implement memory operations
- Add calculation history
- Include unit conversions
