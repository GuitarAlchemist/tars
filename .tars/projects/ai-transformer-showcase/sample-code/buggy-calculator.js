// Sample JavaScript code with intentional bugs for AI analysis
// This will be analyzed by the AI Transformer Showcase

class Calculator {
    constructor() {
        this.history = [];
        this.currentValue = 0;
    }

    // Bug 1: Off-by-one error in array iteration
    calculateTotal(items) {
        let total = 0;
        for (let i = 0; i <= items.length; i++) {  // Should be i < items.length
            total += items[i].price;
        }
        return total;
    }

    // Bug 2: No input validation
    divide(a, b) {
        return a / b;  // Division by zero not handled
    }

    // Bug 3: Potential security issue with eval
    evaluateExpression(expression) {
        return eval(expression);  // Dangerous use of eval
    }

    // Bug 4: Memory leak - history grows indefinitely
    addToHistory(operation, result) {
        this.history.push({
            operation: operation,
            result: result,
            timestamp: new Date()
        });
        // No cleanup of old history entries
    }

    // Bug 5: Incorrect comparison for floating point
    isEqual(a, b) {
        return a == b;  // Should use === and handle floating point precision
    }

    // Bug 6: Async operation without proper error handling
    async fetchExchangeRate(currency) {
        const response = await fetch(`https://api.exchange.com/${currency}`);
        const data = await response.json();  // No error handling
        return data.rate;
    }

    // Bug 7: Improper null/undefined handling
    processUserInput(input) {
        const trimmed = input.trim();
        if (trimmed.length > 0) {
            return parseFloat(trimmed);
        }
        // Returns undefined instead of handling empty input
    }

    // Bug 8: Race condition in async operations
    async performCalculations(operations) {
        const results = [];
        for (const op of operations) {
            results.push(this.processOperation(op));  // Should await
        }
        return results;
    }

    // Bug 9: Inefficient algorithm - O(n²) when O(n) is possible
    findDuplicates(numbers) {
        const duplicates = [];
        for (let i = 0; i < numbers.length; i++) {
            for (let j = i + 1; j < numbers.length; j++) {
                if (numbers[i] === numbers[j] && !duplicates.includes(numbers[i])) {
                    duplicates.push(numbers[i]);
                }
            }
        }
        return duplicates;
    }

    // Bug 10: Incorrect scope and variable hoisting
    calculateCompoundInterest(principal, rate, time) {
        if (principal > 0) {
            var result = principal * Math.pow(1 + rate, time);
        }
        return result;  // 'result' might be undefined
    }
}

// Bug 11: Global variable pollution
var globalCalculator = new Calculator();

// Bug 12: Event listener not properly removed
function setupCalculator() {
    const button = document.getElementById('calculate');
    button.addEventListener('click', function() {
        globalCalculator.performCalculations();
    });
    // No cleanup when component is destroyed
}

// Bug 13: Prototype pollution vulnerability
function mergeObjects(target, source) {
    for (const key in source) {
        target[key] = source[key];  // Can pollute prototype
    }
    return target;
}

// Bug 14: Improper error handling in promises
function calculateAsync(value) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (value > 0) {
                resolve(value * 2);
            } else {
                reject('Invalid value');  // Should use Error object
            }
        }, 1000);
    });
}

// Bug 15: SQL injection vulnerability (if this were server-side)
function getUserCalculations(userId) {
    const query = `SELECT * FROM calculations WHERE user_id = ${userId}`;
    // Direct string interpolation creates SQL injection risk
    return database.query(query);
}

export default Calculator;
