using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using TarsApp.Components;
using TarsApp.Services;
using TarsApp.Services.Interfaces;
using TarsApp.Services.Ingestion;
using TarsApp.ViewModels;
using OllamaSharp;
using MudBlazor.Services;
using TarsEngine.Services.Interfaces;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddRazorComponents().AddInteractiveServerComponents();
builder.Services.AddMudServices(); // Add MudBlazor services

// Use llama3 for both chat and embeddings
IChatClient chatClient = new OllamaApiClient(new Uri("http://localhost:11434"),
    "llama3");
IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator = new OllamaApiClient(new Uri("http://localhost:11434"),
    "llama3"); // Using llama3 for embeddings instead of all-minilm

var vectorStore = new JsonVectorStore(Path.Combine(AppContext.BaseDirectory, "vector-store"));

builder.Services.AddSingleton<IVectorStore>(vectorStore);
builder.Services.AddScoped<DataIngestor>();
builder.Services.AddSingleton<SemanticSearch>();
builder.Services.AddChatClient(chatClient)
    // Remove the UseFunctionInvocation() call since Ollama doesn't support it
    .UseLogging();
builder.Services.AddEmbeddingGenerator(embeddingGenerator);

builder.Services.AddDbContext<IngestionCacheDbContext>(options =>
    options.UseSqlite("Data Source=ingestioncache.db"));

// Register our view model factory
builder.Services.AddSingleton<ViewModelFactory>();

// Register our services
builder.Services.AddScoped<TarsApp.Services.Interfaces.IExecutionPlannerService, TarsApp.Services.ExecutionPlannerService>();
builder.Services.AddScoped<TarsApp.Services.Interfaces.IImprovementPrioritizerService, TarsApp.Services.ImprovementPrioritizerService>();

// Register TarsEngine services (mock implementations for now)
builder.Services.AddScoped<TarsEngine.Services.Interfaces.IExecutionService, MockExecutionService>();
builder.Services.AddScoped<TarsEngine.Services.Interfaces.IImprovementService, MockImprovementService>();

var app = builder.Build();
IngestionCacheDbContext.Initialize(app.Services);

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseAntiforgery();

app.UseStaticFiles();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

// By default, we ingest PDF files from the /wwwroot/Data directory. You can ingest from
// other sources by implementing IIngestionSource.
// Important: ensure that any content you ingest is trusted, as it may be reflected back
// to users or could be a source of prompt injection risk.
await DataIngestor.IngestDataAsync(
    app.Services,
    new PDFDirectorySource(Path.Combine(builder.Environment.WebRootPath, "Data")));

app.Run();
