using Xunit;

public class CalculatorTests
{
    [Fact]
    public void Add_TwoPositiveNumbers_ReturnsSum()
    {
        var calculator = new Calculator();
        double result = calculator.Add(2, 3);
        Assert.Equal(5, result);
    }

    [Fact]
    public void Add_TwoNegativeNumbers_ReturnsSum()
    {
        var calculator = new Calculator();
        double result = calculator.Add(-2, -3);
        Assert.Equal(-5, result);
    }

    [Fact]
    public void Add_OnePositiveOneNegative_ReturnsSum()
    {
        var calculator = new Calculator();
        double result = calculator.Add(2, -3);
        Assert.Equal(-1, result);
    }

    [Fact]
    public void Subtract_TwoPositiveNumbers_ReturnsDifference()
    {
        var calculator = new Calculator();
        double result = calculator.Subtract(5, 3);
        Assert.Equal(2, result);
    }

    [Fact]
    public void Subtract_TwoNegativeNumbers_ReturnsDifference()
    {
        var calculator = new Calculator();
        double result = calculator.Subtract(-5, -3);
        Assert.Equal(-2, result);
    }

    [Fact]
    public void Subtract_OnePositiveOneNegative_ReturnsDifference()
    {
        var calculator = new Calculator();
        double result = calculator.Subtract(5, -3);
        Assert.Equal(8, result);
    }

    [Fact]
    public void Multiply_TwoPositiveNumbers_ReturnsProduct()
    {
        var calculator = new Calculator();
        double result = calculator.Multiply(4, 5);
        Assert.Equal(20, result);
    }

    [Fact]
    public void Multiply_TwoNegativeNumbers_ReturnsProduct()
    {
        var calculator = new Calculator();
        double result = calculator.Multiply(-4, -5);
        Assert.Equal(20, result);
    }

    [Fact]
    public void Multiply_OnePositiveOneNegative_ReturnsProduct()
    {
        var calculator = new Calculator();
        double result = calculator.Multiply(4, -5);
        Assert.Equal(-20, result);
    }

    [Fact]
    public void Divide_TwoPositiveNumbers_ReturnsQuotient()
    {
        var calculator = new Calculator();
        double result = calculator.Divide(10, 2);
        Assert.Equal(5, result);
    }

    [Fact]
    public void Divide_TwoNegativeNumbers_ReturnsQuotient()
    {
        var calculator = new Calculator();
        double result = calculator.Divide(-10, -2);
        Assert.Equal(5, result);
    }

    [Fact]
    public void Divide_PositiveByZero_ThrowsDivisionByZeroException()
    {
        var calculator = new Calculator();
        Assert.Throws<DivisionByZeroException>(() => calculator.Divide(10, 0));
    }

    [Fact]
    public void Divide_NegativeByZero_ThrowsDivisionByZeroException()
    {
        var calculator = new Calculator();
        Assert.Throws<DivisionByZeroException>(() => calculator.Divide(-10, 0));
    }
}