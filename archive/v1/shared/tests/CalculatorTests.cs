using System;
using Xunit;
using SwarmTest;

namespace SwarmTest.Tests
{
    public class CalculatorTests
    {
        [Fact]
        public void Add_ShouldReturnCorrectSum()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Add(2, 3);
            
            // Assert
            Assert.Equal(5, result);
        }
        
        [Fact]
        public void Subtract_ShouldReturnCorrectDifference()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Subtract(5, 3);
            
            // Assert
            Assert.Equal(2, result);
        }
        
        [Fact]
        public void Multiply_ShouldReturnCorrectProduct()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Multiply(2, 3);
            
            // Assert
            Assert.Equal(6, result);
        }
        
        [Fact]
        public void Divide_ShouldReturnCorrectQuotient()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act
            var result = calculator.Divide(6, 3);
            
            // Assert
            Assert.Equal(2, result);
        }
        
        [Fact]
        public void Divide_ShouldThrowExceptionWhenDividingByZero()
        {
            // Arrange
            var calculator = new Calculator();
            
            // Act & Assert
            Assert.Throws<DivideByZeroException>(() => calculator.Divide(6, 0));
        }
    }
}
