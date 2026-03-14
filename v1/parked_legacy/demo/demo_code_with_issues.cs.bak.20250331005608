using System;
using System.Collections.Generic;

namespace DemoCode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // This is a demo program with some issues to be improved
            Console.WriteLine("Hello, World!");

            // Issue 1: Magic numbers
            int timeout = 300;

            // Issue 2: Inefficient string concatenation in loop
            string result = "";
            for (int i = 0; i < 100; i++)
            {
                result += i.ToString();
            }

            // Issue 3: Empty catch block
            try
            {
                int x = int.Parse("abc");
            }
            catch (Exception)
            {
                // Empty catch block
            }

            // Issue 4: Unused variable
            var unusedList = new List<string>();

            Console.WriteLine(result);
            Console.WriteLine($"Timeout is set to {timeout} seconds");
        }
    }
}