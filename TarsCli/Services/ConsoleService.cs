using System.Text;

namespace TarsCli.Services;

/// <summary>
/// Service for console output formatting
/// </summary>
public class ConsoleService
{
    private readonly object _lock = new();

    /// <summary>
    /// Write a header to the console
    /// </summary>
    /// <param name="text">The header text</param>
    public void WriteHeader(string text)
    {
        lock (_lock)
        {
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(new string('=', 80));
            Console.WriteLine($"  {text}");
            Console.WriteLine(new string('=', 80));
            Console.ResetColor();
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Write a subheader to the console
    /// </summary>
    /// <param name="text">The subheader text</param>
    public void WriteSubHeader(string text)
    {
        lock (_lock)
        {
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"  {text}");
            Console.WriteLine($"  {new string('-', text.Length)}");
            Console.ResetColor();
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Write an info message to the console
    /// </summary>
    /// <param name="text">The info text</param>
    public void WriteInfo(string text)
    {
        lock (_lock)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"  {text}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Write a success message to the console
    /// </summary>
    /// <param name="text">The success text</param>
    public void WriteSuccess(string text)
    {
        lock (_lock)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"  ✓ {text}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Write a warning message to the console
    /// </summary>
    /// <param name="text">The warning text</param>
    public void WriteWarning(string text)
    {
        lock (_lock)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  ⚠ {text}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Write an error message to the console
    /// </summary>
    /// <param name="text">The error text</param>
    public void WriteError(string text)
    {
        lock (_lock)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"  ✗ {text}");
            Console.ResetColor();
        }
    }

    /// <summary>
    /// Write a table to the console
    /// </summary>
    /// <param name="headers">The table headers</param>
    /// <param name="rows">The table rows</param>
    public void WriteTable(IEnumerable<string> headers, IEnumerable<IEnumerable<string>> rows)
    {
        lock (_lock)
        {
            var headersList = headers.ToList();
            var rowsList = rows.Select(r => r.ToList()).ToList();
            
            // Calculate column widths
            var columnWidths = new int[headersList.Count];
            for (var i = 0; i < headersList.Count; i++)
            {
                columnWidths[i] = headersList[i].Length;
                foreach (var row in rowsList)
                {
                    if (i < row.Count && row[i].Length > columnWidths[i])
                    {
                        columnWidths[i] = row[i].Length;
                    }
                }
            }
            
            // Write headers
            Console.WriteLine();
            var headerLine = new StringBuilder("  ");
            for (var i = 0; i < headersList.Count; i++)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                headerLine.Append(headersList[i].PadRight(columnWidths[i]));
                if (i < headersList.Count - 1)
                {
                    headerLine.Append(" | ");
                }
            }
            Console.WriteLine(headerLine.ToString());
            Console.ResetColor();
            
            // Write separator
            var separatorLine = new StringBuilder("  ");
            for (var i = 0; i < headersList.Count; i++)
            {
                separatorLine.Append(new string('-', columnWidths[i]));
                if (i < headersList.Count - 1)
                {
                    separatorLine.Append("-+-");
                }
            }
            Console.WriteLine(separatorLine.ToString());
            
            // Write rows
            foreach (var row in rowsList)
            {
                var rowLine = new StringBuilder("  ");
                for (var i = 0; i < headersList.Count; i++)
                {
                    var value = i < row.Count ? row[i] : string.Empty;
                    rowLine.Append(value.PadRight(columnWidths[i]));
                    if (i < headersList.Count - 1)
                    {
                        rowLine.Append(" | ");
                    }
                }
                Console.WriteLine(rowLine.ToString());
            }
            
            Console.WriteLine();
        }
    }
}
