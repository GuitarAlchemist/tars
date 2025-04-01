using Xunit;
using Examples;

public class CalculatorTests
{
    [Fact]
    public void Add_WhenPositiveNumbers_ReturnsSum()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Add(2.5, 3.7);

        // Assert
        Assert.Equal(6.2, result);
    }

    [Fact]
    public void Add_WhenNegativeNumbers_ReturnsSum()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Add(-2.5, -3.7);

        // Assert
        Assert.Equal(-6.2, result);
    }

    [Fact]
    public void Add_WhenOneNegativeNumber_ReturnsSum()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Add(2.5, -3.7);

        // Assert
        Assert.Equal(-1.2, result);
    }

    [Fact]
    public void Subtract_WhenPositiveNumbers_ReturnsDifference()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Subtract(5.5, 3.7);

        // Assert
        Assert.Equal(1.8, result);
    }

    [Fact]
    public void Subtract_WhenNegativeNumbers_ReturnsDifference()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Subtract(-2.5, -3.7);

        // Assert
        Assert.Equal(1.2, result);
    }

    [Fact]
    public void Subtract_WhenOneNegativeNumber_ReturnsDifference()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Subtract(5.5, -3.7);

        // Assert
        Assert.Equal(9.2, result);
    }

    [Fact]
    public void Multiply_WhenPositiveNumbers_ReturnsProduct()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Multiply(4.5, 3.7);

        // Assert
        Assert.Equal(16.65, result);
    }

    [Fact]
    public void Multiply_WhenNegativeNumbers_ReturnsProduct()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Multiply(-4.5, -3.7);

        // Assert
        Assert.Equal(16.65, result);
    }

    [Fact]
    public void Multiply_WhenOneNegativeNumber_ReturnsProduct()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Multiply(4.5, -3.7);

        // Assert
        Assert.Equal(-16.65, result);
    }

    [Fact]
    public void Divide_WhenDivisorIsZero_ReturnsNaN()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Divide(4.5, 0);

        // Assert
        Assert.Throws<InvalidOperationException>(() => calculator.Divide(4.5, 0));
    }

    [Fact]
    public void Divide_WhenPositiveNumbers_ReturnsQuotient()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Divide(9.0, 3.0);

        // Assert
        Assert.Equal(3.0, result);
    }

    [Fact]
    public void Divide_WhenNegativeNumbers_ReturnsQuotient()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Divide(-9.0, -3.0);

        // Assert
        Assert.Equal(3.0, result);
    }

    [Fact]
    public void Divide_WhenOneNegativeNumber_ReturnsQuotient()
    {
        // Arrange
        var calculator = new Calculator();

        // Act
        var result = calculator.Divide(-9.0, 3.0);

        // Assert
        Assert.Equal(-3.0, result);
    }
}