using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for debugging TARS DSL scripts
/// </summary>
public class DslDebugCommand : Command
{
    private readonly DslDebuggerService _debuggerService;
    private readonly ILogger<DslDebugCommand> _logger;
        
    /// <summary>
    /// Constructor
    /// </summary>
    public DslDebugCommand(IServiceProvider serviceProvider)
        : base("debug", "Debug a TARS DSL script")
    {
        _debuggerService = serviceProvider.GetRequiredService<DslDebuggerService>();
        _logger = serviceProvider.GetRequiredService<ILogger<DslDebugCommand>>();
            
        // Add subcommands
        AddCommand(CreateRunCommand());
        AddCommand(CreateBreakpointCommand());
        AddCommand(CreateWatchCommand());
        AddCommand(CreateStepCommand());
    }
        
    private Command CreateRunCommand()
    {
        var command = new Command("run", "Run a DSL file with debugging");
            
        var fileOption = new Option<string>(
            aliases: new[] { "--file", "-f" },
            description: "The path to the DSL file")
        {
            IsRequired = true
        };
            
        command.AddOption(fileOption);
            
        command.SetHandler(async (string file) =>
        {
            var result = await _debuggerService.RunWithDebuggingAsync(file);
            Environment.ExitCode = result;
        }, fileOption);
            
        return command;
    }
        
    private Command CreateBreakpointCommand()
    {
        var command = new Command("breakpoint", "Manage breakpoints");
            
        // Add subcommand
        command.AddCommand(CreateAddBreakpointCommand());
        command.AddCommand(CreateRemoveBreakpointCommand());
        command.AddCommand(CreateListBreakpointsCommand());
        command.AddCommand(CreateClearBreakpointsCommand());
            
        return command;
    }
        
    private Command CreateAddBreakpointCommand()
    {
        var command = new Command("add", "Add a breakpoint");
            
        var fileOption = new Option<string>(
            aliases: new[] { "--file", "-f" },
            description: "The file path")
        {
            IsRequired = true
        };
            
        var lineOption = new Option<int>(
            aliases: new[] { "--line", "-l" },
            description: "The line number")
        {
            IsRequired = true
        };
            
        command.AddOption(fileOption);
        command.AddOption(lineOption);
            
        command.SetHandler((string file, int line) =>
        {
            _debuggerService.AddBreakpoint(file, line);
        }, fileOption, lineOption);
            
        return command;
    }
        
    private Command CreateRemoveBreakpointCommand()
    {
        var command = new Command("remove", "Remove a breakpoint");
            
        var fileOption = new Option<string>(
            aliases: new[] { "--file", "-f" },
            description: "The file path")
        {
            IsRequired = true
        };
            
        var lineOption = new Option<int>(
            aliases: new[] { "--line", "-l" },
            description: "The line number")
        {
            IsRequired = true
        };
            
        command.AddOption(fileOption);
        command.AddOption(lineOption);
            
        command.SetHandler((string file, int line) =>
        {
            _debuggerService.RemoveBreakpoint(file, line);
        }, fileOption, lineOption);
            
        return command;
    }
        
    private Command CreateListBreakpointsCommand()
    {
        var command = new Command("list", "List all breakpoints");
            
        command.SetHandler(() =>
        {
            var breakpoints = _debuggerService.ListBreakpoints();
                
            foreach (var (file, line) in breakpoints)
            {
                Console.WriteLine($"{file}:{line}");
            }
        });
            
        return command;
    }
        
    private Command CreateClearBreakpointsCommand()
    {
        var command = new Command("clear", "Clear all breakpoints");
            
        command.SetHandler(() =>
        {
            _debuggerService.ClearBreakpoints();
        });
            
        return command;
    }
        
    private Command CreateWatchCommand()
    {
        var command = new Command("watch", "Manage variable watches");
            
        // Add subcommands
        command.AddCommand(CreateAddWatchCommand());
        command.AddCommand(CreateRemoveWatchCommand());
        command.AddCommand(CreateListWatchesCommand());
        command.AddCommand(CreateClearWatchesCommand());
            
        return command;
    }
        
    private Command CreateAddWatchCommand()
    {
        var command = new Command("add", "Add a variable watch");
            
        var variableOption = new Option<string>(
            aliases: new[] { "--variable", "-v" },
            description: "The variable name")
        {
            IsRequired = true
        };
            
        command.AddOption(variableOption);
            
        command.SetHandler((string variable) =>
        {
            _debuggerService.AddWatch(variable);
        }, variableOption);
            
        return command;
    }
        
    private Command CreateRemoveWatchCommand()
    {
        var command = new Command("remove", "Remove a variable watch");
            
        var variableOption = new Option<string>(
            aliases: new[] { "--variable", "-v" },
            description: "The variable name")
        {
            IsRequired = true
        };
            
        command.AddOption(variableOption);
            
        command.SetHandler((string variable) =>
        {
            _debuggerService.RemoveWatch(variable);
        }, variableOption);
            
        return command;
    }
        
    private Command CreateListWatchesCommand()
    {
        var command = new Command("list", "List all watched variables");
            
        command.SetHandler(() =>
        {
            var watches = _debuggerService.ListWatches();
                
            foreach (var variable in watches)
            {
                Console.WriteLine(variable);
            }
        });
            
        return command;
    }
        
    private Command CreateClearWatchesCommand()
    {
        var command = new Command("clear", "Clear all watched variables");
            
        command.SetHandler(() =>
        {
            _debuggerService.ClearWatches();
        });
            
        return command;
    }
        
    private Command CreateStepCommand()
    {
        var command = new Command("step", "Control stepping through code");
            
        // Add subcommands
        command.AddCommand(CreateStepNextCommand());
        command.AddCommand(CreateStepIntoCommand());
        command.AddCommand(CreateStepOutCommand());
        command.AddCommand(CreateContinueCommand());
            
        return command;
    }
        
    private Command CreateStepNextCommand()
    {
        var command = new Command("next", "Step to the next line");
            
        command.SetHandler(() =>
        {
            _debuggerService.StepNext();
        });
            
        return command;
    }
        
    private Command CreateStepIntoCommand()
    {
        var command = new Command("into", "Step into a function");
            
        command.SetHandler(() =>
        {
            _debuggerService.StepInto();
        });
            
        return command;
    }
        
    private Command CreateStepOutCommand()
    {
        var command = new Command("out", "Step out of a function");
            
        command.SetHandler(() =>
        {
            _debuggerService.StepOut();
        });
            
        return command;
    }
        
    private Command CreateContinueCommand()
    {
        var command = new Command("continue", "Continue execution");
            
        command.SetHandler(() =>
        {
            _debuggerService.Continue();
        });
            
        return command;
    }
}