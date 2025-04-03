using System;
using System.Reflection;
using FluentValidation;
using FluentValidation.AspNetCore;
using MediatR;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.OpenApi.Models;
using ProductCatalog.Application.Behaviors;
using ProductCatalog.Domain.Repositories;
using ProductCatalog.Infrastructure.Data;
using ProductCatalog.Infrastructure.Repositories;

namespace ProductCatalog.API
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            // Add controllers
            services.AddControllers()
                .AddFluentValidation(fv => 
                {
                    fv.RegisterValidatorsFromAssembly(Assembly.GetExecutingAssembly());
                    fv.RegisterValidatorsFromAssembly(Assembly.GetAssembly(typeof(Application.Commands.Products.CreateProductCommand)));
                });

            // Add API versioning
            services.AddApiVersioning(options =>
            {
                options.ReportApiVersions = true;
                options.AssumeDefaultVersionWhenUnspecified = true;
                options.DefaultApiVersion = new Microsoft.AspNetCore.Mvc.ApiVersion(1, 0);
            });

            // Add Swagger
            services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new OpenApiInfo 
                { 
                    Title = "ProductCatalog API", 
                    Version = "v1",
                    Description = "The Product Catalog Microservice API",
                    Contact = new OpenApiContact
                    {
                        Name = "Development Team",
                        Email = "dev@example.com"
                    }
                });
            });

            // Add MediatR
            services.AddMediatR(cfg => 
            {
                cfg.RegisterServicesFromAssembly(Assembly.GetAssembly(typeof(Application.Commands.Products.CreateProductCommand)));
                cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(ValidationBehavior<,>));
                cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(LoggingBehavior<,>));
                cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(TransactionBehavior<,>));
            });

            // Add AutoMapper
            services.AddAutoMapper(Assembly.GetAssembly(typeof(Application.Mappings.MappingProfile)));

            // Add DbContext
            services.AddDbContext<ProductCatalogContext>(options =>
                options.UseSqlServer(
                    Configuration.GetConnectionString("DefaultConnection"),
                    sqlServerOptionsAction: sqlOptions =>
                    {
                        sqlOptions.MigrationsAssembly(typeof(ProductCatalogContext).Assembly.GetName().Name);
                        sqlOptions.EnableRetryOnFailure(maxRetryCount: 15, maxRetryDelay: TimeSpan.FromSeconds(30), errorNumbersToAdd: null);
                    }));

            // Add repositories
            services.AddScoped<IUnitOfWork, UnitOfWork>();
            services.AddScoped<IProductRepository, ProductRepository>();
            services.AddScoped<ICategoryRepository, CategoryRepository>();

            // Add health checks
            services.AddHealthChecks()
                .AddDbContextCheck<ProductCatalogContext>();

            // Add CORS
            services.AddCors(options =>
            {
                options.AddPolicy("CorsPolicy",
                    builder => builder
                        .SetIsOriginAllowed((host) => true)
                        .AllowAnyMethod()
                        .AllowAnyHeader()
                        .AllowCredentials());
            });
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }
            else
            {
                app.UseExceptionHandler("/Error");
                app.UseHsts();
            }

            app.UseSwagger();
            app.UseSwaggerUI(c =>
            {
                c.SwaggerEndpoint("/swagger/v1/swagger.json", "ProductCatalog API V1");
                c.RoutePrefix = string.Empty;
            });

            app.UseHttpsRedirection();
            app.UseRouting();
            app.UseCors("CorsPolicy");
            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
                endpoints.MapHealthChecks("/health");
            });
        }
    }
}
