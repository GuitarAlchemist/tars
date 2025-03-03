using Tars.Components;
using MudBlazor.Services;
using Blazored.LocalStorage;
using TarsEngine.Services;
using NLog;
using NLog.Web;

var logger = LogManager.Setup()
    .LoadConfigurationFromAppSettings()
    .GetCurrentClassLogger();
try
{
    var builder = WebApplication.CreateBuilder(args);

    builder.Services.AddRazorComponents()
        .AddInteractiveServerComponents();

    builder.Services.AddMudServices();
    builder.Services.AddBlazoredLocalStorage();
    builder.Services.AddScoped<ITarsEngineService, TarsEngineServiceService>();
    builder.Services.AddScoped<ChatBotService>();

    builder.Logging.ClearProviders();
    builder.Logging.AddConsole();
    builder.Logging.AddDebug();

    var app = builder.Build();

    if (!app.Environment.IsDevelopment())
    {
        app.UseExceptionHandler("/Error", createScopeForErrors: true);
        app.UseHsts();
    }

    app.UseHttpsRedirection();
    app.UseStaticFiles();
    app.UseAntiforgery();

    app.MapRazorComponents<App>()
        .AddInteractiveServerRenderMode();

    app.Run();
}
catch (Exception ex)
{
    logger.Error(ex, "Stopped program because of exception");
    throw;
}
finally
{
    LogManager.Shutdown();
}