using System;
using System.CommandLine;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for managing TARS personas
/// </summary>
public class PersonaCommand : TarsCommand
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="PersonaCommand"/> class
    /// </summary>
    /// <param name="serviceProvider">The service provider</param>
    public PersonaCommand(IServiceProvider serviceProvider) 
        : base("persona", "Manage TARS personas")
    {
        _serviceProvider = serviceProvider;
        
        // Add subcommands
        AddCommand(CreateSetCommand());
        AddCommand(CreateInfoCommand());
        AddCommand(CreateListCommand());
    }

    private TarsCommand CreateSetCommand()
    {
        var command = new TarsCommand("set", "Set the active persona");
        var nameOption = new Option<string>("--name", "The name of the persona to set") { IsRequired = true };
        command.AddOption(nameOption);
        
        command.SetHandler((string name) =>
        {
            var personaService = _serviceProvider.GetRequiredService<PersonaService>();
            if (personaService.SetPersona(name))
            {
                CliSupport.WriteColorLine($"Persona set to {personaService.CurrentPersona.Name}", ConsoleColor.Green);
                Console.WriteLine(personaService.GetGreeting());
            }
            else
            {
                CliSupport.WriteColorLine($"Persona '{name}' not found", ConsoleColor.Red);
                Console.WriteLine("Available personas:");
                foreach (var persona in personaService.GetAllPersonas())
                {
                    Console.WriteLine($"- {persona.Name}");
                }
            }
        }, nameOption);
        
        return command;
    }

    private TarsCommand CreateInfoCommand()
    {
        var command = new TarsCommand("info", "Get information about the current persona");
        
        command.SetHandler(() =>
        {
            var personaService = _serviceProvider.GetRequiredService<PersonaService>();
            var persona = personaService.CurrentPersona;
            
            CliSupport.WriteHeader($"Persona: {persona.Name}");
            Console.WriteLine($"Description: {persona.Description}");
            Console.WriteLine($"Humor Level: {persona.HumorLevel:P0}");
            Console.WriteLine($"Formality Level: {persona.Formality:P0}");
            
            var traits = persona.GetTraits();
            if (traits.Count > 4) // Skip the basic traits that are already displayed
            {
                Console.WriteLine("\nAdditional Traits:");
                foreach (var trait in traits.Where(t => 
                    t.Key != "Name" && t.Key != "Description" && 
                    t.Key != "HumorLevel" && t.Key != "Formality"))
                {
                    if (trait.Value is string[] array)
                    {
                        Console.WriteLine($"- {trait.Key}:");
                        foreach (var item in array)
                        {
                            Console.WriteLine($"  * {item}");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"- {trait.Key}: {trait.Value}");
                    }
                }
            }
            
            Console.WriteLine("\nGreeting:");
            Console.WriteLine(persona.GetGreeting());
        });
        
        return command;
    }

    private TarsCommand CreateListCommand()
    {
        var command = new TarsCommand("list", "List all available personas");
        
        command.SetHandler(() =>
        {
            var personaService = _serviceProvider.GetRequiredService<PersonaService>();
            var currentPersona = personaService.CurrentPersona;
            
            CliSupport.WriteHeader("Available Personas");
            
            foreach (var persona in personaService.GetAllPersonas())
            {
                if (persona.Name == currentPersona.Name)
                {
                    CliSupport.WriteColorLine($"* {persona.Name} (Current)", ConsoleColor.Green);
                }
                else
                {
                    Console.WriteLine($"- {persona.Name}");
                }
                Console.WriteLine($"  {persona.Description}");
                Console.WriteLine();
            }
            
            Console.WriteLine("Use 'tarscli persona set --name <persona>' to change the active persona.");
        });
        
        return command;
    }
}
