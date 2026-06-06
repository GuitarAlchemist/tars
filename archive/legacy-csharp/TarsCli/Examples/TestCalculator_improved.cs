namespace TarsCli.Examples;

/// <summary>
/// A simple calculator class for testing retroaction coding
/// </summary>
public class TestCalculatorImproved
{
    // Add two numbers
    public int Add(int a, int b)
    {
        return a + b;
    }

    // Subtract two numbers
    public int Subtract(int a, int b)
    {
        return a - b;
    }

    // Multiply two numbers
    public int Multiply(int a, int b)
    {
        return a * b;
    }

    // Divide two numbers
    public int Divide(int a, int b)
    {
        if (b == 0)
        {
            throw new DivideByZeroException("Cannot divide by zero");
        }
        return a / b;
    }

    // Calculate the average of a list of numbers
    public double Average(List<int> numbers)
    {
        if (numbers == null)
        {
            throw new ArgumentNullException(nameof(numbers));
        }

        return numbers.Average();  // This could throw a DivideByZeroException
    }

    // Find the maximum number in a list
    public int Max(List<int> numbers)
    {
        if (numbers == null)
        {
            throw new ArgumentNullException(nameof(numbers));
        }

        if (numbers.Count == 0)
        {
            throw new ArgumentException("List cannot be empty");
        }

        return numbers.Max();
    }

    // Format a message with a number
    public string FormatMessage(string message, int number)
    {
        if (message == null)
        {
            throw new ArgumentNullException(nameof(message));
        }

        return $"{DateTime.Now}: {number} {message}";
    }
}