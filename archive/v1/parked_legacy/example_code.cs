using System;
using System.Collections.Generic;
using System.Linq;

namespace ExampleCode
{
    public class Calculator
    {
        // This method adds two numbers
        public int Add(int a, int b)
        {
            return a + b;
        }

        // This method subtracts two numbers
        public int Subtract(int a, int b)
        {
            return a - b;
        }

        // This method multiplies two numbers
        public int Multiply(int a, int b)
        {
            return a * b;
        }

        // This method divides two numbers
        public int Divide(int a, int b)
        {
            return a / b;  // This could throw a DivideByZeroException
        }

        // This method calculates the average of a list of numbers
        public double Average(List<int> numbers)
        {
            int sum = 0;
            for (int i = 0; i < numbers.Count; i++)
            {
                sum = sum + numbers[i];
            }
            return sum / numbers.Count;  // This could throw a DivideByZeroException
        }

        // This method finds the maximum number in a list
        public int Max(List<int> numbers)
        {
            if (numbers == null || numbers.Count == 0)
            {
                throw new ArgumentException("List cannot be null or empty");
            }

            int max = numbers[0];
            for (int i = 1; i < numbers.Count; i++)
            {
                if (numbers[i] > max)
                {
                    max = numbers[i];
                }
            }
            return max;
        }

        // This method checks if a number is prime
        public bool IsPrime(int number)
        {
            if (number <= 1) return false;
            if (number == 2) return true;
            if (number % 2 == 0) return false;

            var boundary = (int)Math.Floor(Math.Sqrt(number));
          
            for (int i = 3; i <= boundary; i += 2)
            {
                if (number % i == 0)
                    return false;
            }

            return true;
        }
    }
}
