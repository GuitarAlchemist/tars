namespace AutonomousProject.Controllers

open Microsoft.AspNetCore.Mvc
open AutonomousProject.Domain

[<ApiController>]
[<Route("api/[controller]")>]
type UsersController() =
    inherit ControllerBase()
    
    [<HttpGet>]
    member this.GetUsers() =
        [| 
            { Id = System.Guid.NewGuid(); Name = "John Doe"; Email = "john@example.com"; CreatedAt = System.DateTime.UtcNow }
            { Id = System.Guid.NewGuid(); Name = "Jane Smith"; Email = "jane@example.com"; CreatedAt = System.DateTime.UtcNow }
        |]
    
    [<HttpPost>]
    member this.CreateUser([<FromBody>] request: CreateUserRequest) =
        let user = UserService.createUser request
        this.Ok(user)
