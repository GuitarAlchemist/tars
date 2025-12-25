module Tars.Interface.Cli.ConsoleHelpers

open System

/// Provides ASCII-safe alternatives for Unicode symbols when output is piped/redirected
/// Each property checks Console.IsOutputRedirected at access time
[<AbstractClass; Sealed>]
type Symbols private () =
    /// Check if we should use ASCII mode
    static member private UseAscii = Console.IsOutputRedirected

    // Emoji/Unicode -> ASCII mappings (checked at access time)
    static member puzzle = if Symbols.UseAscii then "[P]" else "🧩"
    static member checkmark = if Symbols.UseAscii then "[OK]" else "✅"
    static member cross = if Symbols.UseAscii then "[X]" else "❌"
    static member warning = if Symbols.UseAscii then "[!]" else "⚠️"
    static member lightning = if Symbols.UseAscii then "[*]" else "⚡"
    static member fire = if Symbols.UseAscii then "[~]" else "🔥"
    static member rocket = if Symbols.UseAscii then "[>>]" else "🚀"
    static member flag = if Symbols.UseAscii then "[#]" else "🏁"
    static member globe = if Symbols.UseAscii then "[@]" else "🌐"
    static member plug = if Symbols.UseAscii then "[>]" else "🔌"
    static member satellite = if Symbols.UseAscii then "[^]" else "📡"
    static member stopwatch = if Symbols.UseAscii then "[T]" else "⏱️"
    static member wrench = if Symbols.UseAscii then "[W]" else "🔧"
    static member memo = if Symbols.UseAscii then "[M]" else "📝"
    static member question = if Symbols.UseAscii then "[?]" else "❓"
    static member folder = if Symbols.UseAscii then "[D]" else "📁"
    static member info = if Symbols.UseAscii then "[i]" else "ℹ️"

