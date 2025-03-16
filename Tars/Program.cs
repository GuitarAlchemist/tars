using Blazored.LocalStorage;
using MudBlazor.Services;
using NLog.Web;
using Tars.Components;
using TarsEngine.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Add SignalR with increased timeout for better stability
builder.Services.AddSignalR(options =>
{
    options.ClientTimeoutInterval = TimeSpan.FromSeconds(60);
    options.KeepAliveInterval = TimeSpan.FromSeconds(30);
});

builder.Services.AddMudServices();
builder.Services.AddBlazoredLocalStorage();

// Add logging
builder.Logging.ClearProviders();
builder.Host.UseNLog();

// Add custom services
builder.Services.AddScoped<RivaWrapperService>();
builder.Services.AddScoped<WebSpeechService>();
builder.Services.AddScoped<SpeechServiceFactory>();
builder.Services.AddScoped<ChatBotService>();
builder.Services.AddScoped<ITarsEngineService, TarsEngineServiceService>();

var app = builder.Build();

// Configure the HTTP request pipeline
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
