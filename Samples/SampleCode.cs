using System;
using System.Collections.Generic;
using System.Linq;

namespace Samples
{
    /// <summary>
    /// A sample class with some code issues for testing the Tree-of-Thought reasoning.
    /// </summary>
    public class SampleCode
    {
        // Unused variable
        private int unusedVariable = 42;
        
        // Magic number
        public int CalculateTotal(int quantity)
        {
            return quantity * 10;
        }
        
        // Missing null check
        public string GetFirstItem(List<string> items)
        {
            return items[0];
        }
        
        // Inefficient LINQ
        public List<int> GetEvenNumbers(List<int> numbers)
        {
            return numbers.Where(n => n % 2 == 0).ToList();
        }
        
        // Empty catch block
        public void ProcessData(string data)
        {
            try
            {
                int value = int.Parse(data);
                Console.WriteLine(value);
            }
            catch (Exception)
            {
                // Swallowing the exception
            }
        }
        
        // Long method
        public void DoManyThings()
        {
            Console.WriteLine("Step 1");
            // ... imagine 50 more lines of code here
            Console.WriteLine("Step 2");
            // ... imagine 50 more lines of code here
            Console.WriteLine("Step 3");
            // ... imagine 50 more lines of code here
            Console.WriteLine("Done");
        }
        
        // Complex condition
        public bool IsValidUser(User user)
        {
            if (user != null && user.Name != null && user.Name.Length > 0 && user.Age > 0 && user.Age < 120 && user.Email != null && user.Email.Contains("@") && user.Email.Contains("."))
            {
                return true;
            }
            
            return false;
        }
    }
    
    public class User
    {
        public string Name { get; set; }
        public int Age { get; set; }
        public string Email { get; set; }
    }
}
