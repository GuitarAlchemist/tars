using System;
using System.Collections.Generic;

namespace TarsCli.Examples
{
    /// <summary>
    /// Example class for demonstrating auto-coding
    /// </summary>
    public class AutoCodingExample
    {
        // Method to add two numbers
/// <summary>
/// Method Add
/// </summary>
/// <param name="a">The a parameter</param>
/// <param name="b">The b parameter</param>
/// <returns>The result of the operation</returns>
        public int Add(int a, int b)
        {
            return a + b;
        }

        // Method to subtract two numbers with a bug
/// <summary>
/// Method Subtract
/// </summary>
/// <param name="a">The a parameter</param>
/// <param name="b">The b parameter</param>
/// <returns>The result of the operation</returns>
        public int Subtract(int a, int b)
        {
            return a - b + 1; // Bug: adding 1 unnecessarily
        }

        // Method to multiply two numbers with poor naming
/// <summary>
/// Method x
/// </summary>
/// <param name="a">The a parameter</param>
/// <param name="b">The b parameter</param>
/// <returns>The result of the operation</returns>
        public int x(int a, int b)
        {
            return a * b;
        }

        // Method to divide two numbers without error handling
/// <summary>
/// Method Divide
/// </summary>
/// <param name="a">The a parameter</param>
/// <param name="b">The b parameter</param>
/// <returns>The result of the operation</returns>
        public int Divide(int a, int b)
        {
            return a / b; // Missing error handling for division by zero
        }

        // Method to calculate the average with inefficient implementation
/// <summary>
/// Method Average
/// </summary>
/// <param name="numbers">The numbers parameter</param>
/// <returns>The result of the operation</returns>
        public double Average(List<int> numbers)
        {
            int sum = 0;
            for (int i = 0; i < numbers.Count; i++)
            {
                sum = sum + numbers[i];
            }
            return sum / numbers.Count; // Missing error handling for empty list
        }

        // Method to find maximum with unnecessary complexity
/// <summary>
/// Method FindMax
/// </summary>
/// <param name="numbers">The numbers parameter</param>
/// <returns>The result of the operation</returns>
        public int FindMax(List<int> numbers)
        {
            if (numbers == null || numbers.Count == 0)
                throw new ArgumentException("List cannot be null or empty");

            int max = int.MinValue;
            foreach (var num in numbers)
            {
                if (num > max)
                    max = num;
            }
            return max;
        }

        // Method to find minimum with a bug
/// <summary>
/// Method FindMin
/// </summary>
/// <param name="numbers">The numbers parameter</param>
/// <returns>The result of the operation</returns>
        public int FindMin(List<int> numbers)
        {
            if (numbers == null || numbers.Count == 0)
                throw new ArgumentException("List cannot be null or empty");

            int min = int.MaxValue;
            foreach (var num in numbers)
            {
                if (num < min)
                    min = num;
            }
            return min + 1; // Bug: adding 1 unnecessarily
        }
    }
}
