using System;
using Xunit;
using SwarmTest;

namespace SwarmTest.Tests
{
    public class StringUtilsTests
    {
        [Fact]
        public void Reverse_ShouldReturnReversedString()
        {
            // Act
            var result = StringUtils.Reverse("hello");
            
            // Assert
            Assert.Equal("olleh", result);
        }
        
        [Fact]
        public void Reverse_ShouldHandleEmptyString()
        {
            // Act
            var result = StringUtils.Reverse("");
            
            // Assert
            Assert.Equal("", result);
        }
        
        [Fact]
        public void IsPalindrome_ShouldReturnTrueForPalindrome()
        {
            // Act
            var result = StringUtils.IsPalindrome("racecar");
            
            // Assert
            Assert.True(result);
        }
        
        [Fact]
        public void IsPalindrome_ShouldReturnFalseForNonPalindrome()
        {
            // Act
            var result = StringUtils.IsPalindrome("hello");
            
            // Assert
            Assert.False(result);
        }
        
        [Fact]
        public void CountWords_ShouldReturnCorrectWordCount()
        {
            // Act
            var result = StringUtils.CountWords("hello world");
            
            // Assert
            Assert.Equal(2, result);
        }
        
        [Fact]
        public void Truncate_ShouldTruncateString()
        {
            // Act
            var result = StringUtils.Truncate("hello world", 8);
            
            // Assert
            Assert.Equal("hello...", result);
        }
        
        [Fact]
        public void Truncate_ShouldNotTruncateShortString()
        {
            // Act
            var result = StringUtils.Truncate("hello", 10);
            
            // Assert
            Assert.Equal("hello", result);
        }
    }
}
