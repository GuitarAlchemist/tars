using TarsEngine.Functional.Examples;
using TarsEngine.Monads;

namespace TarsEngine.Examples;

/// <summary>
/// User class for validation examples
/// </summary>
public class User
{
    public string Username { get; }
    public string Password { get; }
    public int Age { get; }

    public User(string username, string password, int age)
    {
        Username = username;
        Password = password;
        Age = age;
    }

    public override string ToString() => $"User {{ Username = {Username}, Password = {Password}, Age = {Age} }}";
}

/// <summary>
/// Examples of how to use functional programming patterns in C#
/// </summary>
public static class FunctionalProgrammingExamples
{
    #region EnhancedOption Examples

    /// <summary>
    /// Example of using the EnhancedOption monad to handle nullable references
    /// </summary>
    public static void EnhancedOptionExample()
    {
        // Create options
        var someValue = EnhancedOption.Some("Hello, World!");
        var noneValue = EnhancedOption.None<string>();

        // Pattern matching
        var greeting = someValue.Match(
            some: value => $"Value: {value}",
            none: () => "No value"
        );
        Console.WriteLine(greeting); // Output: Value: Hello, World!

        // Chaining operations
        var result = someValue
            .Map(s => s.Length)
            .Filter(length => length > 10)
            .ValueOr(0);
        Console.WriteLine($"Result: {result}"); // Output: Result: 13

        // Working with collections
        var optionalValues = new[] { 1, 2, 3, 4, 5 }
            .Select(n => n % 2 == 0
                ? EnhancedOption.Some(n)
                : EnhancedOption.None<int>());

        var evenNumbers = optionalValues.Choose().ToList();
        Console.WriteLine($"Even numbers: {string.Join(", ", evenNumbers)}"); // Output: Even numbers: 2, 4

        // Dictionary lookup
        var dictionary = new Dictionary<string, int>
        {
            ["one"] = 1,
            ["two"] = 2,
            ["three"] = 3
        };

        var lookupResult = dictionary.TryGetValue("four")
            .ValueOr(-1);
        Console.WriteLine($"Lookup result: {lookupResult}"); // Output: Lookup result: -1

        // Parsing
        var parseResult = "123".TryParseInt()
            .Map(n => n * 2)
            .ValueOr(0);
        Console.WriteLine($"Parse result: {parseResult}"); // Output: Parse result: 246

        // Combining options
        var option1 = EnhancedOption.Some(10);
        var option2 = EnhancedOption.Some(20);
        var option3 = EnhancedOption.Some(30);

        var combined = EnhancedOption.Map3(
            option1, option2, option3,
            (a, b, c) => a + b + c
        );
        Console.WriteLine($"Combined: {combined.ValueOr(0)}"); // Output: Combined: 60
    }

    #endregion

    #region EnhancedAsyncResult Examples

    /// <summary>
    /// Example of using the EnhancedAsyncResult monad to handle asynchronous operations
    /// </summary>
    public static async Task EnhancedAsyncResultExampleAsync()
    {
        // Create async results
        var asyncResult1 = EnhancedAsyncResult.FromResult(10);
        var asyncResult2 = EnhancedAsyncResult.FromTask(Task.FromResult(20));
        var asyncResult3 = EnhancedAsyncResult.TryAsync(async () => {
            await Task.Delay(100);
            return 30;
        });

        // Chaining operations
        var result = await asyncResult1
            .Map(n => n * 2)
            .Bind(n => EnhancedAsyncResult.FromResult(n + 5))
            .Do(n => Console.WriteLine($"Intermediate result: {n}"))
            .RunAsync();
        Console.WriteLine($"Result: {result}"); // Output: Intermediate result: 25, Result: 25

        // Combining async results
        var combined = await EnhancedAsyncResult.Map3(
            asyncResult1, asyncResult2, asyncResult3,
            (a, b, c) => a + b + c
        ).RunAsync();
        Console.WriteLine($"Combined: {combined}"); // Output: Combined: 60

        // Working with collections
        var asyncResults = Enumerable.Range(1, 5)
            .Select(n => EnhancedAsyncResult.FromResult(n * 10));

        var sequence = await EnhancedAsyncResult.Sequence(asyncResults).RunAsync();
        Console.WriteLine($"Sequence: {string.Join(", ", sequence)}"); // Output: Sequence: 10, 20, 30, 40, 50

        // Error handling with AsyncResultError
        var successResult = EnhancedAsyncResult.Success<int, string>(42);
        var failureResult = EnhancedAsyncResult.Failure<int, string>("Something went wrong");

        var successMessage = await successResult.RunAsync()
            .ContinueWith(t => t.Result.Match(
                success: value => $"Success: {value}",
                failure: error => $"Failure: {error}"
            ));
        Console.WriteLine(successMessage); // Output: Success: 42

        var failureMessage = await failureResult.RunAsync()
            .ContinueWith(t => t.Result.Match(
                success: value => $"Success: {value}",
                failure: error => $"Failure: {error}"
            ));
        Console.WriteLine(failureMessage); // Output: Failure: Something went wrong
    }

    #endregion

    #region Either Examples

    /// <summary>
    /// Example of using the Either discriminated union to handle success/failure cases
    /// </summary>
    public static void EitherExample()
    {
        // Create Either values
        var right = Either<string, int>.Right(42);
        var left = Either<string, int>.Left("Something went wrong");

        // Pattern matching
        var rightResult = right.Match(
            leftFunc: error => $"Error: {error}",
            rightFunc: value => $"Value: {value}"
        );
        Console.WriteLine(rightResult); // Output: Value: 42

        var leftResult = left.Match(
            leftFunc: error => $"Error: {error}",
            rightFunc: value => $"Value: {value}"
        );
        Console.WriteLine(leftResult); // Output: Error: Something went wrong

        // Chaining operations
        var chainedRight = right
            .Map(n => n * 2)
            .Bind(n => Either<string, int>.Right(n + 5));
        Console.WriteLine($"Chained right: {chainedRight.RightValue}"); // Output: Chained right: 89

        var chainedLeft = left
            .Map(n => n * 2)
            .Bind(n => Either<string, int>.Right(n + 5));
        Console.WriteLine($"Chained left: {chainedLeft.LeftValue}"); // Output: Chained left: Something went wrong

        // Try operation
        var trySuccess = EitherExtensions.Try(() => 10 / 2);
        var tryFailure = EitherExtensions.Try(() => {
            // This would cause a division by zero exception
            // but it's safely handled by the Try method
            var divisor = 0;
            return 10 / divisor;
        });

        Console.WriteLine($"Try success: {trySuccess.RightValueOrDefault()}"); // Output: Try success: 5
        Console.WriteLine($"Try failure: {tryFailure.LeftValue.Message}"); // Output: Try failure: Attempted to divide by zero.

        // Converting between Either and Option
        var option = right.ToOption();
        var eitherFromOption = option.ToEither("Option was None");

        Console.WriteLine($"Option: {option.ValueOrDefault}"); // Output: Option: 42
        Console.WriteLine($"Either from option: {eitherFromOption.RightValue}"); // Output: Either from option: 42
    }

    #endregion

    #region Validation Examples

    /// <summary>
    /// Example of using the Validation discriminated union to validate inputs
    /// </summary>
    public static void ValidationExample()
    {
        // Define validation rules
        Validation<string, string> ValidateUsername(string username) =>
            string.IsNullOrWhiteSpace(username)
                ? Validation<string, string>.Invalid("Username cannot be empty")
                : username.Length < 3
                    ? Validation<string, string>.Invalid("Username must be at least 3 characters")
                    : username.Length > 20
                        ? Validation<string, string>.Invalid("Username cannot be longer than 20 characters")
                        : Validation<string, string>.Valid(username);

        Validation<string, string> ValidatePassword(string password) =>
            string.IsNullOrWhiteSpace(password)
                ? Validation<string, string>.Invalid("Password cannot be empty")
                : password.Length < 8
                    ? Validation<string, string>.Invalid("Password must be at least 8 characters")
                    : !password.Any(char.IsUpper)
                        ? Validation<string, string>.Invalid("Password must contain at least one uppercase letter")
                        : !password.Any(char.IsDigit)
                            ? Validation<string, string>.Invalid("Password must contain at least one digit")
                            : Validation<string, string>.Valid(password);

        Validation<int, string> ValidateAge(int age) =>
            age < 18
                ? Validation<int, string>.Invalid("You must be at least 18 years old")
                : age > 120
                    ? Validation<int, string>.Invalid("Invalid age")
                    : Validation<int, string>.Valid(age);

        // Use the User class defined at the namespace level

        // Validate user input
        var usernameValidation = ValidateUsername("john_doe");
        var passwordValidation = ValidatePassword("Password123");
        var ageValidation = ValidateAge(25);

        // Combine validations
        var userValidation = ValidationExtensions.Map3(
            usernameValidation,
            passwordValidation,
            ageValidation,
            (username, password, age) => new User(username, password, age)
        );

        // Handle validation result
        var validationResult = userValidation.Match(
            validFunc: user => $"Valid user: {user}",
            invalidFunc: errors => $"Validation errors: {string.Join(", ", errors)}"
        );
        Console.WriteLine(validationResult);
        // Output: Valid user: User { Username = john_doe, Password = Password123, Age = 25 }

        // Example with validation errors
        var invalidUsernameValidation = ValidateUsername("jo");
        var invalidPasswordValidation = ValidatePassword("pass");
        var invalidAgeValidation = ValidateAge(15);

        var invalidUserValidation = ValidationExtensions.Map3(
            invalidUsernameValidation,
            invalidPasswordValidation,
            invalidAgeValidation,
            (username, password, age) => new User(username, password, age)
        );

        var invalidValidationResult = invalidUserValidation.Match(
            validFunc: user => $"Valid user: {user}",
            invalidFunc: errors => $"Validation errors: {string.Join(", ", errors)}"
        );
        Console.WriteLine(invalidValidationResult);
        // Output: Validation errors: Username must be at least 3 characters, Password must be at least 8 characters, Password must contain at least one uppercase letter, Password must contain at least one digit, You must be at least 18 years old

        // Validating collections
        var items = new[] { "apple", "b", "banana", "", "cherry" };
        var itemValidations = items.Select(item =>
            string.IsNullOrWhiteSpace(item)
                ? Validation<string, string>.Invalid("Item cannot be empty")
                : item.Length < 2
                    ? Validation<string, string>.Invalid($"Item '{item}' is too short")
                    : Validation<string, string>.Valid(item)
        );

        var validatedItems = itemValidations.Sequence();
        var itemsValidationResult = validatedItems.Match(
            validFunc: validItems => $"Valid items: {string.Join(", ", validItems)}",
            invalidFunc: errors => $"Validation errors: {string.Join(", ", errors)}"
        );
        Console.WriteLine(itemsValidationResult);
        // Output: Validation errors: Item 'b' is too short, Item cannot be empty
    }

    #endregion

    #region Combined Examples

    /// <summary>
    /// Example of combining multiple functional programming patterns
    /// </summary>
    public static async Task CombinedExampleAsync()
    {
        // Define a function that might return null
        EnhancedOption<string> GetUsername(int userId) =>
            userId switch
            {
                1 => EnhancedOption.Some("john_doe"),
                2 => EnhancedOption.Some("jane_smith"),
                _ => EnhancedOption.None<string>()
            };

        // Define a function that might fail
        async Task<Result<int, Exception>> GetUserAgeAsync(string username)
        {
            await Task.Delay(100); // Simulate API call
            return username switch
            {
                "john_doe" => Result<int, Exception>.Success(25),
                "jane_smith" => Result<int, Exception>.Success(30),
                _ => Result<int, Exception>.Failure(new ArgumentException($"User not found: {username}"))
            };
        }

        // Define a validation function
        Validation<int, string> ValidateAge(int age) =>
            age < 18
                ? Validation<int, string>.Invalid("You must be at least 18 years old")
                : age > 120
                    ? Validation<int, string>.Invalid("Invalid age")
                    : Validation<int, string>.Valid(age);

        // Combine all patterns
        var userId = 1;

        // 1. Get the username (Option)
        var usernameOption = GetUsername(userId);

        // 2. Convert to Either for error handling
        var usernameEither = usernameOption.ToEither($"User not found with ID: {userId}");

        // 3. Use AsyncResultError for async operation
        var ageResult = await usernameEither.Match(
            leftFunc: error => Task.FromResult(Result<int, Exception>.Failure(new Exception(error))),
            rightFunc: username => GetUserAgeAsync(username)
        );

        // 4. Convert to Validation for validation
        var ageValidation = ageResult.Match(
            success: age => ValidateAge(age),
            failure: ex => Validation<int, string>.Invalid(ex.Message)
        );

        // 5. Handle the final result
        var finalResult = ageValidation.Match<string>(
            validFunc: age => $"User is {age} years old and eligible",
            invalidFunc: errors => $"Validation failed: {string.Join(", ", errors)}"
        );
        Console.WriteLine(finalResult);
        // Output: User is 25 years old and eligible
    }

    #endregion
}