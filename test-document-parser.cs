using System;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

class Program
{
    static void Main(string[] args)
    {
        // Test the line ending fix
        string content = "Console.WriteLine(\"Hello, World!\");";
        string content2 = "Console.WriteLine(\"Hello, World!\");\r";
        
        Console.WriteLine("Original content: " + BitConverter.ToString(Encoding.UTF8.GetBytes(content)));
        Console.WriteLine("Content with CR: " + BitConverter.ToString(Encoding.UTF8.GetBytes(content2)));
        
        Console.WriteLine("After TrimEnd('\\r'):");
        Console.WriteLine("Original content: " + BitConverter.ToString(Encoding.UTF8.GetBytes(content.TrimEnd('\r'))));
        Console.WriteLine("Content with CR: " + BitConverter.ToString(Encoding.UTF8.GetBytes(content2.TrimEnd('\r'))));
        
        Console.WriteLine("Equal after TrimEnd? " + (content.TrimEnd('\r') == content2.TrimEnd('\r')));
        
        // Test the code block regex
        string markdownContent = "```csharp\nConsole.WriteLine(\"Hello, World!\");\n```\n\n```fsharp\nprintfn \"Hello, World!\"\n```";
        var regex = new Regex(@"```(?<language>[^\n]*)\n(?<code>.*?)```", RegexOptions.Singleline);
        
        var matches = regex.Matches(markdownContent);
        Console.WriteLine($"\nFound {matches.Count} code blocks");
        
        foreach (Match match in matches)
        {
            Console.WriteLine($"Language: {match.Groups["language"].Value}");
            Console.WriteLine($"Code: {match.Groups["code"].Value}");
            Console.WriteLine();
        }
    }
}
