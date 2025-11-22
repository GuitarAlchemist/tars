namespace AutonomousProject.Domain

open System

type User = {
    Id: Guid
    Name: string
    Email: string
    CreatedAt: DateTime
}

type CreateUserRequest = {
    Name: string
    Email: string
}

module UserService =
    let createUser (request: CreateUserRequest) : User =
        {
            Id = Guid.NewGuid()
            Name = request.Name
            Email = request.Email
            CreatedAt = DateTime.UtcNow
        }
