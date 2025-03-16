// Ensure you add the correct F# library reference providing TarsEngine functionality

using System;
using System.Threading;
using System.Threading.Tasks;
using TarsEngine;
using Microsoft.FSharp.Control;
using Microsoft.FSharp.Core;
using Microsoft.Extensions.Logging;
using TarsEngineFSharp;

namespace TarsEngine.Services;

public class ChatBotService
{
    private TarsEngineFSharp.ChatService.ChatSession _session;
    private readonly ILogger<ChatBotService> _logger;

    public ChatBotService(ILogger<ChatBotService> logger)
    {
        _session = TarsEngineFSharp.ChatService.createNewSession();
        _logger = logger;
    }

    public async Task<ChatResponse> GetResponse(string message)
    {
        _logger.LogInformation("GetResponse called with message: {Message}", message);
        
        try 
        {
            _session = TarsEngineFSharp.ChatService.addMessage(_session, "user", message);
            
            _logger.LogDebug("Processing message through ChatService");
            var response = await FSharpAsync.StartAsTask(
                TarsEngineFSharp.ChatService.processMessage(message),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None
            );

            _session = TarsEngineFSharp.ChatService.addMessage(_session, "assistant", response.Text);
            
            _logger.LogInformation("Response received: {Response}", response.Text);
            return new ChatResponse(response.Text, response.Source);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing message: {Message}", message);
            throw;
        }
    }
}

public record ChatResponse(string Text, string Source);