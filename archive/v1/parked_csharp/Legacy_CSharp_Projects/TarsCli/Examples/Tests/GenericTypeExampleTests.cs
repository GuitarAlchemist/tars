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
    /// Tests for the GenericTypeExample class
    /// </summary>
    [TestClass]
    public class GenericTypeExampleTests
    {
        /// <summary>
        /// Tests the Average method
        /// </summary>
        [TestMethod]
        public void Average_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.Average(numbers);

            // Assert
            Assert.AreEqual(20.0, result, 0.001);
        }

        /// <summary>
        /// Tests the FindMax method
        /// </summary>
        [TestMethod]
        public void FindMax_ShouldFind_Item_When_Exists()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.FindMax(numbers);

            // Assert
            Assert.AreEqual(30, result);
        }

        /// <summary>
        /// Tests the FindMin method
        /// </summary>
        [TestMethod]
        public void FindMin_ShouldFind_Item_When_Exists()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.FindMin(numbers);

            // Assert
            Assert.AreEqual(10, result);
        }

        /// <summary>
        /// Tests the FilterEven method
        /// </summary>
        [TestMethod]
        public void FilterEven_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.FilterEven(numbers);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(3, result.Count);
            CollectionAssert.AreEqual(new List<int> { 10, 20, 30 }, result);
        }

        /// <summary>
        /// Tests the DictionaryToList method
        /// </summary>
        [TestMethod]
        public void DictionaryToList_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var dictionary = new Dictionary<string, int>() { { "test1", 10 }, { "test2", 20 } };

            // Act
            var result = sut.DictionaryToList(dictionary);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count);
            CollectionAssert.Contains(result, "test1: 10");
            CollectionAssert.Contains(result, "test2: 20");
        }

        /// <summary>
        /// Tests the MergeDictionaries method
        /// </summary>
        [TestMethod]
        public void MergeDictionaries_ShouldReturn_Expected_Value()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var first = new Dictionary<string, int>() { { "test1", 10 }, { "test2", 20 } };
            var second = new Dictionary<string, int>() { { "test1", 10 }, { "test2", 20 } };

            // Act
            var result = sut.MergeDictionaries(first, second);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count);
            Assert.AreEqual(20, result["test1"]);
            Assert.AreEqual(40, result["test2"]);
        }

    }
}
