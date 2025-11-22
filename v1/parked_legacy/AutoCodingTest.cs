// This is a test file for auto-coding
// The code has been improved by TARS

using System;

namespace AutoCodingTest
{
    public class Calculator
    {
        /// <summary>
        /// Adds two numbers and returns the result.
        /// </summary>
        /// <param name="a">First number</param>
        /// <param name="b">Second number</param>
        /// <returns>The sum of a and b</returns>
        public double Add(double a, double b)
        {
            return a + b;
        }
        
        /// <summary>
        /// Subtracts the second number from the first and returns the result.
        /// </summary>
        /// <param name="a">First number</param>
        /// <param name="b">Second number</param>
        /// <returns>The difference between a and b</returns>
        public double Subtract(double a, double b)
        {
            return a - b;
        }
        
        /// <summary>
        /// Multiplies two numbers and returns the result.
        /// </summary>
        /// <param name="a">First number</param>
        /// <param name="b">Second number</param>
        /// <returns>The product of a and b</returns>
        public double Multiply(double a, double b)
        {
            return a * b;
        }
        
        /// <summary>
        /// Divides the first number by the second and returns the result.
        /// </summary>
        /// <param name="a">First number (dividend)</param>
        /// <param name="b">Second number (divisor)</param>
        /// <returns>The quotient of a divided by b</returns>
        /// <exception cref="DivideByZeroException">Thrown when b is zero</exception>
        public double Divide(double a, double b)
        {
            if (b == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero");
            }
            
            return a / b;
        }
    }
}
