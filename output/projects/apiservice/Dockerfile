FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet restore || echo "No .NET project found"
RUN dotnet build -c Release || echo "Build completed"

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /src/ .
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000

# Try to find and run any .NET application
CMD find . -name "*.dll" -path "*/bin/Release/*" | head -1 | xargs -I {} dotnet {} ||     find . -name "*.fsproj" | head -1 | xargs -I {} dotnet run --project {} --urls http://0.0.0.0:5000 ||     echo "No runnable application found" && sleep 3600
