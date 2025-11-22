using System;
using System.Collections.Generic;

namespace TarsEngine.Test
{
    /// <summary>
    /// A test class with code quality issues for the auto-improvement pipeline to fix.
    /// </summary>
    public class TestClass
    {
        // Unused variable
        private int unusedVariable = 42;

        // Missing null check
        public void ProcessData(string data)
        {
            Console.WriteLine(data.Length);
        }

        // Inefficient LINQ
        public List<int> GetEvenNumbers(List<int> numbers)
        {
            var result = new List<int>();
            foreach (var number in numbers)
            {
                if (number % 2 == 0)
                {
                    result.Add(number);
                }
            }
            return result;
        }

        // Magic number
        public double CalculateCircleArea(double radius)
        {
            return 3.14159 * radius * radius;
        }

        // Empty catch block
        public void RiskyOperation()
        {
            try
            {
                // Some risky operation
                int.Parse("not a number");
            }
            catch (Exception)
            {
                // Empty catch block
            }
        }
    }
}
