using System.CommandLine;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for working with TARS DSL files
/// </summary>
public class DslCommand : Command
{
    private readonly DslService _dslService;

    public DslCommand(DslService dslService) : base("dsl", "Work with TARS DSL files")
    {
        _dslService = dslService;

        // Add subcommands
        AddCommand(CreateRunCommand());
        AddCommand(CreateValidateCommand());
        AddCommand(CreateGenerateCommand());
        AddCommand(CreateEbnfCommand());
    }

    private Command CreateRunCommand()
    {
        var command = new Command("run", "Run a TARS DSL file");

        var fileOption = new Option<string>(
            aliases: new[] { "--file", "-f" },
            description: "The path to the DSL file to run")
        {
            IsRequired = true
        };

        var verboseOption = new Option<bool>(
            aliases: new[] { "--verbose", "-v" },
            description: "Show verbose output");

        command.AddOption(fileOption);
        command.AddOption(verboseOption);

        command.SetHandler(async (string file, bool verbose) =>
        {
            var result = await _dslService.RunDslFileAsync(file, verbose);
            Environment.ExitCode = result;
        }, fileOption, verboseOption);

        return command;
    }

    private Command CreateValidateCommand()
    {
        var command = new Command("validate", "Validate a TARS DSL file");

        var fileOption = new Option<string>(
            aliases: new[] { "--file", "-f" },
            description: "The path to the DSL file to validate")
        {
            IsRequired = true
        };

        command.AddOption(fileOption);

        command.SetHandler(async (string file) =>
        {
            var result = await _dslService.ValidateDslFileAsync(file);
            Environment.ExitCode = result;
        }, fileOption);

        return command;
    }

    private Command CreateGenerateCommand()
    {
        var command = new Command("generate", "Generate a TARS DSL file template");

        var outputOption = new Option<string>(
            aliases: new[] { "--output", "-o" },
            description: "The path to save the generated DSL file")
        {
            IsRequired = true
        };

        var templateOption = new Option<string>(
            aliases: new[] { "--template", "-t" },
            description: "The template to use (basic, chat, agent)")
        {
            IsRequired = false
        };

        templateOption.SetDefaultValue("basic");

        command.AddOption(outputOption);
        command.AddOption(templateOption);

        command.SetHandler(async (string output, string template) =>
        {
            var result = await _dslService.GenerateDslTemplateAsync(output, template);
            Environment.ExitCode = result;
        }, outputOption, templateOption);

        return command;
    }

    private Command CreateEbnfCommand()
    {
        var command = new Command("ebnf", "Generate EBNF for the TARS DSL");

        var outputOption = new Option<string>(
            aliases: new[] { "--output", "-o" },
            description: "The path to save the EBNF file")
        {
            IsRequired = false
        };

        outputOption.SetDefaultValue("tars_dsl.ebnf");

        command.AddOption(outputOption);

        command.SetHandler(async (string output) =>
        {
            var result = await _dslService.GenerateEbnfAsync(output);
            Environment.ExitCode = result;
        }, outputOption);

        return command;
    }
}