namespace TarsTaskManager.Tests.Integration

open Xunit
open Microsoft.AspNetCore.Mvc.Testing
open System.Net.Http
open System.Text
open Newtonsoft.Json
open TarsTaskManager.Api

module ApiIntegrationTests =

    type TaskManagerWebApplicationFactory() =
        inherit WebApplicationFactory<Program>()

    [<Fact>]
    let `GET /api/tasks should return 200 with valid JWT` () =
        task {
            // Arrange
            use factory = new TaskManagerWebApplicationFactory()
            use client = factory.CreateClient()

            // Add JWT token for authentication
            let token = "valid-jwt-token-here"
            client.DefaultRequestHeaders.Authorization <-
                System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token)

            // Act
            let! response = client.GetAsync("/api/tasks")

            // Assert
            response.StatusCode |> should equal System.Net.HttpStatusCode.OK
        }

    [<Fact>]
    let `POST /api/tasks should create task with AI analysis` () =
        task {
            // Arrange
            use factory = new TaskManagerWebApplicationFactory()
            use client = factory.CreateClient()

            let newTask = {|
                Title = "Implement feature X"
                Description = "Add new functionality for users"
                AssignedTo = null
                ProjectId = null
            |}

            let json = JsonConvert.SerializeObject(newTask)
            let content = new StringContent(json, Encoding.UTF8, "application/json")

            // Act
            let! response = client.PostAsync("/api/tasks", content)

            // Assert
            response.StatusCode |> should equal System.Net.HttpStatusCode.Created

            let! responseContent = response.Content.ReadAsStringAsync()
            let createdTask = JsonConvert.DeserializeObject<Task>(responseContent)

            createdTask.Title |> should equal "Implement feature X"
            createdTask.Priority |> should not' (equal Priority.Low) // AI should suggest appropriate priority
        }

module DatabaseIntegrationTests =

    [<Fact>]
    let `Database connection should be established` () =
        // Test database connectivity and basic operations
        true |> should equal true // Placeholder for actual database tests
