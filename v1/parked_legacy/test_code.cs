using System;
using System.Collections.Generic;
using System.Text;

namespace TestCode
{
    public class Program
    {
        // Constants for magic numbers
        private const int DefaultTimeout = 300;
        private const int LoopCount = 100;

        public static void Main(string[] args)
        {
            // This is a test program with some issues to be improved
            Console.WriteLine("Hello, World!");

            // Issue 1: Magic numbers - Fixed by using constants
            int timeout = DefaultTimeout;

            // Issue 2: Inefficient string concatenation in loop - Fixed by using StringBuilder
            StringBuilder resultBuilder = new StringBuilder();
            for (int i = 0; i < LoopCount; i++)
            {
                resultBuilder.Append(i.ToString());
            }
            string result = resultBuilder.ToString();

            // Issue 3: Empty catch block - Fixed by adding logging
            try
            {
                int x = int.Parse("abc");
            }
            catch (Exception ex)
            {
                // Log the exception instead of having an empty catch block
                Console.WriteLine($"Error parsing string: {ex.Message}");
            }

            // Issue 4: Unused variable - Fixed by removing or using it
            var dataList = new List<string>();
            dataList.Add("Sample data");
            Console.WriteLine($"Data count: {dataList.Count}");

            Console.WriteLine(result);
            Console.WriteLine($"Timeout is set to {timeout} seconds");
        }
    }
}