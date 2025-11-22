using System;
using System.Collections.Generic;

namespace Examples
{
    public class TestFile
    {
        // This is a test file for the self-improvement feature
        
        public void BadMethod()
        {
            // This method has some issues that could be improved
            
            // Issue 1: Unused variable
            int unused = 10;
            
            // Issue 2: Magic number
            int result = 42 * 2;
            
            // Issue 3: Inefficient string concatenation
            string message = "";
            for (int i = 0; i < 10; i++)
            {
                message = message + i.ToString();
            }
            
            // Issue 4: Not using var for obvious types
            List<string> items = new List<string>();
            
            // Issue 5: Not using null check
            string input = null;
            if (input == null)
            {
                Console.WriteLine("Input is null");
            }
            
            // Issue 6: Not using string interpolation
            string name = "TARS";
            string greeting = "Hello, " + name + "!";
            
            Console.WriteLine(greeting);
        }
        
        public int Calculate(int a, int b)
        {
            // Issue 7: Not checking for division by zero
            return a / b;
        }
    }
}
