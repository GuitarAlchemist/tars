using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using TarsCliMinimal.Examples;

namespace TarsCliMinimal.Examples.Tests
{
    [TestClass]
    public class GenericTypeExampleTests
    {
        [TestMethod]
        public void Test_Average_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.Average(numbers);

            // Assert
            Assert.AreEqual(20.0, result, 0.001, "Average should be 20.0");
        }

        [TestMethod]
        public void Test_FindMax_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.FindMax(numbers);

            // Assert
            Assert.AreEqual(30, result, "Max should be 30");
        }

        [TestMethod]
        public void Test_CountOccurrences_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var dictionary = new Dictionary<string, int>() { { "key1", 10 }, { "key2", 20 } };

            // Act
            var result = sut.CountOccurrences(dictionary, "key1");

            // Assert
            Assert.AreEqual(1, result, "Count should be 1");
        }

        [TestMethod]
        public void Test_ConvertToList_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var dictionary = new Dictionary<string, int>() { { "key1", 10 }, { "key2", 20 } };

            // Act
            var result = sut.ConvertToList(dictionary);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
            Assert.AreEqual(2, result.Count, "List should have 2 items");
            Assert.AreEqual("key1", result[0].Key, "First key should be 'key1'");
            Assert.AreEqual(10, result[0].Value, "First value should be 10");
        }

        [TestMethod]
        public void Test_MergeLists_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var list1 = new List<int>() { 10, 20 };
            var list2 = new List<int>() { 30, 40 };

            // Act
            var result = sut.MergeLists(list1, list2);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
            Assert.AreEqual(4, result.Count, "List should have 4 items");
            CollectionAssert.AreEqual(new List<int>() { 10, 20, 30, 40 }, result, "Lists should be merged correctly");
        }

        [TestMethod]
        public void Test_FilterList_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var list = new List<int>() { 10, 20, 30, 40 };
            Func<int, bool> predicate = x => x > 20;

            // Act
            var result = sut.FilterList(list, predicate);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
            Assert.AreEqual(2, result.Count, "List should have 2 items");
            CollectionAssert.AreEqual(new List<int>() { 30, 40 }, result, "List should be filtered correctly");
        }

        [TestMethod]
        public void Test_CreateDictionary_ShouldWork()
        {
            // Arrange
            var sut = new GenericTypeExample();
            var keys = new List<string>() { "key1", "key2" };
            var values = new List<int>() { 10, 20 };

            // Act
            var result = sut.CreateDictionary(keys, values);

            // Assert
            Assert.IsNotNull(result, "Result should not be null");
            Assert.AreEqual(2, result.Count, "Dictionary should have 2 items");
            Assert.AreEqual(10, result["key1"], "Value for 'key1' should be 10");
            Assert.AreEqual(20, result["key2"], "Value for 'key2' should be 20");
        }
    }
}
