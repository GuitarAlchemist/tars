\nusing System;\nusing System.Collections.Generic;\nusing System.Text;\n\nnamespace DemoCode\n{\n    public class Program\n    {\n        private const int MaxItems = 100;\n        private const int TimeoutSeconds = 300;\n\n        public static void Main(string[] args)\n        {\n            Console.WriteLine(\"Hello, World!\");\n\n            string result = new StringBuilder().Insert(0, \"Result: \").ToString();\n            for (int i = 0; i \u003c MaxItems; i++)\n            {\n                result = new StringBuilder(result).AppendLine(i.ToString()).ToString();\n            }\n\n            try\n            {\n                int x = int.Parse(\"123\");\n            }\n            catch (FormatException)\n            {\n                Console.WriteLine(\"Invalid input. Please enter a valid integer.\");\n            }\n\n            var usedList = new List\u003cstring\u003e { \"Used item 1\", \"Used item 2\" };\n\n            Console.WriteLine(result);\n            Console.WriteLine($\"Timeout is set to {TimeoutSeconds} seconds\");\n        }\n    }\n}\n