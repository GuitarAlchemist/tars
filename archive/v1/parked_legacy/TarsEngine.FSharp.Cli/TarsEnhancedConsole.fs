
// TARS Auto-Generated: Enhanced Console Output
module TarsEnhancedConsole =
    open System
    
    let writeSuccess (message: string) =
        Console.ForegroundColor <- ConsoleColor.Green
        Console.WriteLine($"✅ {message}")
        Console.ResetColor()
    
    let writeWarning (message: string) =
        Console.ForegroundColor <- ConsoleColor.Yellow
        Console.WriteLine($"⚠️  {message}")
        Console.ResetColor()
    
    let writeError (message: string) =
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine($"❌ {message}")
        Console.ResetColor()
    
    let writeInfo (message: string) =
        Console.ForegroundColor <- ConsoleColor.Cyan
        Console.WriteLine($"ℹ️  {message}")
        Console.ResetColor()
