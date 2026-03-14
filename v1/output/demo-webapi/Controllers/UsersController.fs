namespace UserAPI.Controllers

open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open System.Threading.Tasks

[<ApiController>]
[<Route("api/[controller]")>]
type UsersController(logger: ILogger<UsersController>) =
    inherit ControllerBase()

    [<HttpGET("/api/users")>]
    member _.GetUsers(): Task<IActionResult> =
        task {
            logger.LogInformation("Executing GetUsers")
            return Ok("Response from GetUsers")
        }

    [<HttpGET("/api/users/{id}")>]
    member _.GetUser(): Task<IActionResult> =
        task {
            logger.LogInformation("Executing GetUser")
            return Ok("Response from GetUser")
        }

    [<HttpPOST("/api/users")>]
    member _.CreateUser(): Task<IActionResult> =
        task {
            logger.LogInformation("Executing CreateUser")
            return Ok("Response from CreateUser")
        }

    [<HttpPUT("/api/users/{id}")>]
    member _.UpdateUser(): Task<IActionResult> =
        task {
            logger.LogInformation("Executing UpdateUser")
            return Ok("Response from UpdateUser")
        }

    [<HttpDELETE("/api/users/{id}")>]
    member _.DeleteUser(): Task<IActionResult> =
        task {
            logger.LogInformation("Executing DeleteUser")
            return Ok("Response from DeleteUser")
        }