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
            var numbers = new List<int>() { 10, 20, 30 };

            // Act
            var result = sut.Average(numbers);

            // Assert
            Assert.AreEqual(20.0, result, 0.001);
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
            Assert.AreEqual(30, result);
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
            Assert.AreEqual(1, result);
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
            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count);
            Assert.AreEqual("key1", result[0].Key);
            Assert.AreEqual(10, result[0].Value);
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
            Assert.IsNotNull(result);
            Assert.AreEqual(4, result.Count);
            Assert.AreEqual(10, result[0]);
            Assert.AreEqual(20, result[1]);
            Assert.AreEqual(30, result[2]);
            Assert.AreEqual(40, result[3]);
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
            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count);
            Assert.AreEqual(30, result[0]);
            Assert.AreEqual(40, result[1]);
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
            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count);
            Assert.AreEqual(10, result["key1"]);
            Assert.AreEqual(20, result["key2"]);
        }
    }
}
