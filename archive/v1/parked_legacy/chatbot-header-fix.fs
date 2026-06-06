    member private self.ShowChatbotHeader() =
        AnsiConsole.Clear()
        
        let headerPanel = Panel("""[bold cyan]🤖 TARS Interactive Chatbot[/]
[dim]Powered by Mixture of Experts AI System[/]

[bold magenta]💡 Just ask naturally! TARS will route your request to the right expert.[/]
[yellow]Type '[green]help[/]' to see available commands or '[green]exit[/]' to quit.[/]""")
        headerPanel.Header <- PanelHeader("[bold blue]🚀 TARS AI Assistant[/]")
        headerPanel.Border <- BoxBorder.Double
        headerPanel.BorderStyle <- Style.Parse("cyan")
        AnsiConsole.Write(headerPanel)
        AnsiConsole.WriteLine()
