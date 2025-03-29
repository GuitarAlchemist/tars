using System.CommandLine.Parsing;
using System.Text;

namespace TarsCli.Parsing;

/// <summary>
/// Custom argument parser that handles triple-quoted strings (""" """)
/// </summary>
public class TripleQuotedArgumentParser
{
    /// <summary>
    /// Parses command line arguments and combines triple-quoted strings
    /// </summary>
    public static string[] ParseTripleQuotedArguments(string[] args)
    {
        var result = new List<string>();
        bool inTripleQuotedString = false;
        var tripleQuotedBuilder = new StringBuilder();
        
        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            
            if (inTripleQuotedString)
            {
                // Check if this argument ends the triple-quoted string
                if (arg.EndsWith("\"\"\""))
                {
                    // Add the content to the builder and end the triple-quoted string
                    tripleQuotedBuilder.Append(arg.Substring(0, arg.Length - 3));
                    result.Add(tripleQuotedBuilder.ToString());
                    tripleQuotedBuilder.Clear();
                    inTripleQuotedString = false;
                }
                else
                {
                    // Add the content to the builder with a newline
                    tripleQuotedBuilder.AppendLine(arg);
                }
            }
            else if (arg.StartsWith("\"\"\""))
            {
                // Check if this argument also ends the triple-quoted string
                if (arg.EndsWith("\"\"\"") && arg.Length > 6)
                {
                    // This is a complete triple-quoted string in a single argument
                    result.Add(arg.Substring(3, arg.Length - 6));
                }
                else
                {
                    // Start a new triple-quoted string
                    tripleQuotedBuilder.Append(arg.Substring(3));
                    tripleQuotedBuilder.AppendLine();
                    inTripleQuotedString = true;
                }
            }
            else
            {
                // Regular argument
                result.Add(arg);
            }
        }
        
        // If we're still in a triple-quoted string, add what we have so far
        if (inTripleQuotedString)
        {
            result.Add(tripleQuotedBuilder.ToString());
        }
        
        return result.ToArray();
    }
    
    /// <summary>
    /// Processes a command line with triple-quoted strings
    /// </summary>
    public static ParseResult ParseCommandLine(Parser parser, string[] args)
    {
        // First, preprocess the arguments to handle triple-quoted strings
        var processedArgs = ParseTripleQuotedArguments(args);
        
        // Then, parse the processed arguments with the regular parser
        return parser.Parse(processedArgs);
    }
}
