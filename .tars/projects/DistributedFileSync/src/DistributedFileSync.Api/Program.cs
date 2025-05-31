using DistributedFileSync.Core.Interfaces;
using DistributedFileSync.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.OpenApi.Models;
using Serilog;

// Configure Serilog
Log.Logger = new LoggerConfiguration()
    .WriteTo.Console()
    .WriteTo.File("logs/filesync-.txt", rollingInterval: RollingInterval.Day)
    .CreateLogger();

var builder = WebApplication.CreateBuilder(args);

// Add Serilog
builder.Host.UseSerilog();

// Add services to the container
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();

// Configure Swagger/OpenAPI
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "Distributed File Sync API",
        Version = "v1.0.0",
        Description = "RESTful API for Distributed File Synchronization System",
        Contact = new OpenApiContact
        {
            Name = "TARS Multi-Agent Development Team",
            Email = "team@tars-agents.dev"
        }
    });

    // Add JWT authentication to Swagger
    c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Description = "JWT Authorization header using the Bearer scheme",
        Name = "Authorization",
        In = ParameterLocation.Header,
        Type = SecuritySchemeType.ApiKey,
        Scheme = "Bearer"
    });

    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {
            new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference
                {
                    Type = ReferenceType.SecurityScheme,
                    Id = "Bearer"
                }
            },
            Array.Empty<string>()
        }
    });
});

// Add JWT Authentication
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        // JWT configuration would go here
        // Configured by: Security Specialist Agent (Eve)
    });

// Add authorization
builder.Services.AddAuthorization();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// Register application services
// These would be properly configured with dependency injection
builder.Services.AddScoped<ISynchronizationEngine, SynchronizationEngine>();

// Add health checks
builder.Services.AddHealthChecks();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "Distributed File Sync API v1");
        c.RoutePrefix = string.Empty; // Serve Swagger UI at root
    });
}

app.UseHttpsRedirection();
app.UseCors("AllowAll");
app.UseAuthentication();
app.UseAuthorization();

// Add security headers (Security Specialist Agent - Eve)
app.Use(async (context, next) =>
{
    context.Response.Headers.Add("X-Content-Type-Options", "nosniff");
    context.Response.Headers.Add("X-Frame-Options", "DENY");
    context.Response.Headers.Add("X-XSS-Protection", "1; mode=block");
    context.Response.Headers.Add("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
    await next();
});

app.MapControllers();
app.MapHealthChecks("/health");

// Add startup banner
app.Logger.LogInformation("""
    🚀 DISTRIBUTED FILE SYNC API STARTING
    =====================================
    
    📊 System Information:
       • Version: 1.0.0
       • Environment: {Environment}
       • Port: {Port}
       • Swagger UI: {SwaggerUrl}
    
    👥 Developed by TARS Multi-Agent Team:
       • 🏗️ Architect: System design and architecture
       • 💻 Senior Developer: Core implementation
       • 🔬 Researcher: Technology evaluation
       • ⚡ Performance Engineer: Optimization
       • 🛡️ Security Specialist: Security hardening
       • 🤝 Project Coordinator: Team coordination
       • 🧪 QA Engineer: Quality assurance
    
    ✅ API Ready for Distributed File Synchronization!
    """, 
    app.Environment.EnvironmentName,
    builder.Configuration["ASPNETCORE_URLS"] ?? "https://localhost:5001",
    app.Environment.IsDevelopment() ? "https://localhost:5001" : "Disabled in production"
);

try
{
    app.Run();
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application terminated unexpectedly");
}
finally
{
    Log.CloseAndFlush();
}
