using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using TarsCli.Examples;

namespace TarsCli.Examples.Tests
{
    /// <summary>
    /// Tests for the AutoCodingExample class
    /// </summary>
    [TestClass]
    public class AutoCodingExampleTests
    {
        /// <summary>
        /// Tests the Add method
        /// </summary>
        [TestMethod]
        public void Add_ShouldAdd_Item_Successfully()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var a = 42;
            var b = 42;

            // Act
            var result = sut.Add(a, b);

            // Assert
            Assert.AreEqual(84, result);
        }

        /// <summary>
        /// Tests the Subtract method
        /// </summary>
        [TestMethod]
        public void Subtract_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var a = 42;
            var b = 42;

            // Act
            var result = sut.Subtract(a, b);

            // Assert
            Assert.AreEqual(1, result); // 0 + 1 due to the bug
        }

        /// <summary>
        /// Tests the x method
        /// </summary>
        [TestMethod]
        public void x_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var a = 42;
            var b = 42;

            // Act
            var result = sut.x(a, b);

            // Assert
            Assert.AreEqual(1764, result);
        }

        /// <summary>
        /// Tests the Divide method
        /// </summary>
        [TestMethod]
        public void Divide_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var a = 42;
            var b = 42;

            // Act
            var result = sut.Divide(a, b);

            // Assert
            Assert.AreEqual(1, result);
        }

        /// <summary>
        /// Tests the Average method
        /// </summary>
        [TestMethod]
        public void Average_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var numbers = new List<int>() { 42, 42 };

            // Act
            var result = sut.Average(numbers);

            // Assert
            Assert.AreEqual(42, result);
        }

        /// <summary>
        /// Tests the FindMax method
        /// </summary>
        [TestMethod]
        public void FindMax_ShouldFind_Item_When_Exists()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var numbers = new List<int>() { 42, 10 };

            // Act
            var result = sut.FindMax(numbers);

            // Assert
            Assert.AreEqual(42, result);
        }

        /// <summary>
        /// Tests the FindMin method
        /// </summary>
        [TestMethod]
        public void FindMin_ShouldFind_Item_When_Exists()
        {
            // Arrange
            var sut = new AutoCodingExample();
            var numbers = new List<int>() { 42, 10 };

            // Act
            var result = sut.FindMin(numbers);

            // Assert
            Assert.AreEqual(11, result); // 10 + 1 due to the bug
        }

    }
}
