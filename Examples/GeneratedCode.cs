using System;

namespace Examples
{
    public class Calculator
    {
        /// <summary>
        /// Adds two numbers together.
        /// </summary>
        /// <param name="num1">The first number.</param>
        /// <param name="num2">The second number.</param>
        /// <returns>The sum of the two numbers.</returns>
        public double Add(double num1, double num2)
        {
            return num1 + num2;
        }

        /// <summary>
        /// Subtracts one number from another.
        /// </summary>
        /// <param name="num1">The first number.</param>
        /// <param name="num2">The second number.</param>
        /// <returns>The difference of the two numbers.</returns>
        public double Subtract(double num1, double num2)
        {
            return num1 - num2;
        }

        /// <summary>
        /// Multiplies two numbers together.
        /// </summary>
        /// <param name="num1">The first number.</param>
        /// <param name="num2">The second number.</param>
        /// <returns>The product of the two numbers.</returns>
        public double Multiply(double num1, double num2)
        {
            return num1 * num2;
        }

        /// <summary>
        /// Divides one number by another.
        /// </summary>
        /// <param name="num1">The dividend.</param>
        /// <param name="num2">The divisor.</param>
        /// <returns>The quotient of the two numbers, or NaN if num2 is zero.</returns>
        public double Divide(double num1, double num2)
        {
            if (num2 == 0)
                return double.NaN;
            else
                return num1 / num2;
        }
    }

    public class TestFile
    {
        // Add test code here...
    }
}