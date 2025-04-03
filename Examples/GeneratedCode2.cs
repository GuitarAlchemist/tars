Here is the generated code:

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
        /// <returns>The result of the addition.</returns>
        public double Add(double num1, double num2)
        {
            return num1 + num2;
        }

        /// <summary>
        /// Subtracts one number from another.
        /// </summary>
        /// <param name="num1">The first number.</param>
        /// <param name="num2">The second number.</param>
        /// <returns>The result of the subtraction.</returns>
        public double Subtract(double num1, double num2)
        {
            return num1 - num2;
        }

        /// <summary>
        /// Multiplies two numbers together.
        /// </summary>
        /// <param name="num1">The first number.</param>
        /// <param name="num2">The second number.</param>
        /// <returns>The result of the multiplication.</returns>
        public double Multiply(double num1, double num2)
        {
            return num1 * num2;
        }

        /// <summary>
        /// Divides one number by another.
        /// </summary>
        /// <param name="num1">The dividend.</param>
        /// <param name="num2">The divisor.</param>
        /// <returns>The result of the division. Throws a DivisionByZeroException if the divisor is zero.</returns>
        public double Divide(double num1, double num2)
        {
            if (num2 == 0)
            {
                throw new DivisionByZeroException("Cannot divide by zero.");
            }

            return num1 / num2;
        }
    }
}