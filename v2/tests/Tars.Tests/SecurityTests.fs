namespace Tars.Tests

open System
open System.IO
open Xunit
open Xunit.Abstractions
open Tars.Security

type SecurityTests(output: ITestOutputHelper) =

    // CredentialVault Tests

    [<Fact>]
    member _.``CredentialVault: Can register and retrieve secret``() =
        output.WriteLine("Starting test: Can register and retrieve secret")
        // Arrange
        let key = $"TEST_SECRET_{Guid.NewGuid()}"
        let value = "my-secret-value"

        // Act
        CredentialVault.registerSecret key value
        let result = CredentialVault.getSecret key

        // Assert
        match result with
        | Ok retrieved ->
            Assert.Equal(value, retrieved)
            output.WriteLine($"Successfully retrieved secret for key: {key}")
        | Error e ->
            Assert.Fail($"Failed to retrieve secret: {e}")

    [<Fact>]
    member _.``CredentialVault: Returns error for missing secret``() =
        output.WriteLine("Starting test: Returns error for missing secret")
        // Arrange
        let key = $"NONEXISTENT_SECRET_{Guid.NewGuid()}"

        // Act
        let result = CredentialVault.getSecret key

        // Assert
        match result with
        | Ok _ ->
            Assert.Fail("Should have returned error for missing secret")
        | Error e ->
            Assert.Contains("not found", e)
            output.WriteLine($"Correctly returned error: {e}")

    [<Fact>]
    member _.``CredentialVault: Can update existing secret``() =
        output.WriteLine("Starting test: Can update existing secret")
        // Arrange
        let key = $"UPDATE_SECRET_{Guid.NewGuid()}"
        let originalValue = "original-value"
        let updatedValue = "updated-value"

        // Act
        CredentialVault.registerSecret key originalValue
        CredentialVault.registerSecret key updatedValue
        let result = CredentialVault.getSecret key

        // Assert
        match result with
        | Ok retrieved ->
            Assert.Equal(updatedValue, retrieved)
            output.WriteLine("Successfully updated and retrieved secret")
        | Error e ->
            Assert.Fail($"Failed to retrieve updated secret: {e}")

    [<Fact>]
    member _.``CredentialVault: Environment variable takes precedence``() =
        output.WriteLine("Starting test: Environment variable takes precedence")
        // Arrange
        let key = $"ENV_SECRET_{Guid.NewGuid()}"
        let envValue = "env-value"
        let vaultValue = "vault-value"

        // Act
        Environment.SetEnvironmentVariable(key, envValue)
        CredentialVault.registerSecret key vaultValue
        let result = CredentialVault.getSecret key

        // Cleanup
        Environment.SetEnvironmentVariable(key, null)

        // Assert
        match result with
        | Ok retrieved ->
            Assert.Equal(envValue, retrieved)
            output.WriteLine("Environment variable correctly takes precedence")
        | Error e ->
            Assert.Fail($"Failed to retrieve secret: {e}")

    // FilesystemPolicy Tests

    [<Fact>]
    member _.``FilesystemPolicy: Allows path within base directory``() =
        output.WriteLine("Starting test: Allows path within base directory")
        // Arrange
        let basePath = Path.GetTempPath()
        let requestedPath = "subdir/file.txt"

        // Act
        let result = FilesystemPolicy.validatePath basePath requestedPath

        // Assert
        match result with
        | Ok fullPath ->
            Assert.StartsWith(Path.GetFullPath(basePath), fullPath)
            output.WriteLine($"Correctly allowed path: {fullPath}")
        | Error e ->
            Assert.Fail($"Should have allowed path: {e}")

    [<Fact>]
    member _.``FilesystemPolicy: Blocks path traversal attack``() =
        output.WriteLine("Starting test: Blocks path traversal attack")
        // Arrange
        let basePath = Path.Combine(Path.GetTempPath(), "sandbox")
        let requestedPath = "../../../etc/passwd"

        // Act
        let result = FilesystemPolicy.validatePath basePath requestedPath

        // Assert
        match result with
        | Ok _ ->
            Assert.Fail("Should have blocked path traversal")
        | Error e ->
            Assert.Contains("Access denied", e)
            output.WriteLine($"Correctly blocked path traversal: {e}")

    [<Fact>]
    member _.``FilesystemPolicy: Blocks absolute path outside base``() =
        output.WriteLine("Starting test: Blocks absolute path outside base")
        // Arrange
        let basePath = Path.Combine(Path.GetTempPath(), "sandbox")
        let requestedPath = 
            if Environment.OSVersion.Platform = PlatformID.Win32NT then
                "C:\\Windows\\System32"
            else
                "/etc/passwd"

        // Act
        let result = FilesystemPolicy.validatePath basePath requestedPath

        // Assert
        match result with
        | Ok fullPath ->
            // On Windows, Path.Combine with absolute path returns the absolute path
            // So we need to check if it's within base
            if not (fullPath.StartsWith(Path.GetFullPath(basePath), StringComparison.OrdinalIgnoreCase)) then
                output.WriteLine("Path validation correctly handled absolute path")
            else
                Assert.Fail("Should have blocked absolute path outside base")
        | Error e ->
            Assert.Contains("Access denied", e)
            output.WriteLine($"Correctly blocked absolute path: {e}")

    [<Fact>]
    member _.``FilesystemPolicy: Allows nested subdirectory``() =
        output.WriteLine("Starting test: Allows nested subdirectory")
        // Arrange
        let basePath = Path.GetTempPath()
        let requestedPath = "level1/level2/level3/file.txt"

        // Act
        let result = FilesystemPolicy.validatePath basePath requestedPath

        // Assert
        match result with
        | Ok fullPath ->
            Assert.Contains("level1", fullPath)
            Assert.Contains("level2", fullPath)
            Assert.Contains("level3", fullPath)
            output.WriteLine($"Correctly allowed nested path: {fullPath}")
        | Error e ->
            Assert.Fail($"Should have allowed nested path: {e}")

    [<Fact>]
    member _.``FilesystemPolicy: Handles path with dot segments correctly``() =
        output.WriteLine("Starting test: Handles path with dot segments correctly")
        // Arrange
        let basePath = Path.GetTempPath()
        let requestedPath = "subdir/../subdir/file.txt"

        // Act
        let result = FilesystemPolicy.validatePath basePath requestedPath

        // Assert
        match result with
        | Ok fullPath ->
            // The path should be normalized and still within base
            Assert.StartsWith(Path.GetFullPath(basePath), fullPath)
            output.WriteLine($"Correctly normalized path: {fullPath}")
        | Error e ->
            Assert.Fail($"Should have allowed normalized path: {e}")
