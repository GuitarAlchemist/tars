using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsTestGenerator.Examples.Tests
{
    [TestClass]
    public class GenericTypeExampleTests
    {
        [TestMethod]
        public void Test_Average_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 42, 42, 42 };

            // Act
            var result = sut.Average(numbers);

            // Assert
            Assert.IsTrue(!double.IsNaN((double)result), "Average should not be NaN");
        }

        [TestMethod]
        public void Test_FindMax_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 42, 42, 42 };

            // Act
            var result = sut.FindMax(numbers);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
        }

        [TestMethod]
        public void Test_CountOccurrences_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var dictionary = new Dictionary<TKey, TValue>() { { new TKey(), new TValue() }, { new TKey(), new TValue() } };
            var key = new TKey();

            // Act
            var result = sut.CountOccurrences(dictionary, key);

            // Assert
            Assert.IsTrue(result >= 0, "Count should be non-negative");
        }

        [TestMethod]
        public void Test_ConvertToList_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var dictionary = new Dictionary<string, int>() { { "test", 42 }, { "test", 42 } };

            // Act
            var result = sut.ConvertToList(dictionary);

            // Assert
            Assert.IsNotNull(result, "Converted result should not be null");
        }

        [TestMethod]
        public void Test_MergeLists_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var list1 = new List<T>() { new T(), new T(), new T() };
            var list2 = new List<T>() { new T(), new T(), new T() };

            // Act
            var result = sut.MergeLists(list1, list2);

            // Assert
            Assert.IsNotNull(result, "Result collection should not be null");
        }

        [TestMethod]
        public void Test_FilterList_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var list = new List<T>() { new T(), new T(), new T() };
            var predicate = new Func<T, bool>();

            // Act
            var result = sut.FilterList(list, predicate);

            // Assert
            Assert.IsNotNull(result, "Result collection should not be null");
        }

        [TestMethod]
        public void Test_CreateDictionary_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var keys = new List<TKey>() { new TKey(), new TKey(), new TKey() };
            var values = new List<TValue>() { new TValue(), new TValue(), new TValue() };

            // Act
            var result = sut.CreateDictionary(keys, values);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
        }

    }
}
