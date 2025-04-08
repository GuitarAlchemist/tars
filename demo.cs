using System;

namespace DuplicationDemo
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            // Duplicated code block 1
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine($"The sum of {a} and {b} is {c}");

            // Some other code
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
            }

            // Duplicated code block 2
            int x = 1;
            int y = 2;
            int z = x + y;
            Console.WriteLine($"The sum of {x} and {y} is {z}");

            // Semantically similar code
            var first = 10;
            var second = 20;
            var result = first + second;
            Console.WriteLine($"Adding {first} and {second} gives {result}");
        }

        static void AnotherMethod()
        {
            // Duplicated code block 3
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine($"The sum of {a} and {b} is {c}");
        }
    }
}