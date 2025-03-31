using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TarsCli.Common;
using Xunit;

namespace TarsCli.Tests.Services
{
    public class OptionTests
    {
        [Fact]
        public void Some_WithValue_ReturnsOptionWithValue()
        {
            // Arrange
            string value = "test";

            // Act
            var option = Option.Some(value);

            // Assert
            Assert.True(option.IsSome);
            Assert.False(option.IsNone);
            Assert.Equal(value, option.Value);
        }

        [Fact]
        public void None_ReturnsEmptyOption()
        {
            // Act
            var option = Option.None<string>();

            // Assert
            Assert.False(option.IsSome);
            Assert.True(option.IsNone);
            Assert.Throws<InvalidOperationException>(() => option.Value);
        }

        [Fact]
        public void Map_WithSome_TransformsValue()
        {
            // Arrange
            var option = Option.Some(5);

            // Act
            var result = option.Map(x => x * 2);

            // Assert
            Assert.True(result.IsSome);
            Assert.Equal(10, result.Value);
        }

        [Fact]
        public void Map_WithNone_ReturnsNone()
        {
            // Arrange
            var option = Option.None<int>();

            // Act
            var result = option.Map(x => x * 2);

            // Assert
            Assert.True(result.IsNone);
        }

        [Fact]
        public async Task MapAsync_WithSome_TransformsValueAsync()
        {
            // Arrange
            var option = Option.Some(5);

            // Act
            var result = await option.MapAsync(x => Task.FromResult(x * 2));

            // Assert
            Assert.True(result.IsSome);
            Assert.Equal(10, result.Value);
        }

        [Fact]
        public async Task MapAsync_WithNone_ReturnsNoneAsync()
        {
            // Arrange
            var option = Option.None<int>();

            // Act
            var result = await option.MapAsync(x => Task.FromResult(x * 2));

            // Assert
            Assert.True(result.IsNone);
        }

        [Fact]
        public void Bind_WithSome_AppliesBinder()
        {
            // Arrange
            var option = Option.Some(5);

            // Act
            var result = option.Bind(x => x % 2 == 0 ? Option.Some(x / 2) : Option.None<int>());

            // Assert
            Assert.True(result.IsNone);
        }

        [Fact]
        public void Bind_WithNone_ReturnsNone()
        {
            // Arrange
            var option = Option.None<int>();

            // Act
            var result = option.Bind(x => x % 2 == 0 ? Option.Some(x / 2) : Option.None<int>());

            // Assert
            Assert.True(result.IsNone);
        }

        [Fact]
        public void ValueOr_WithSome_ReturnsValue()
        {
            // Arrange
            var option = Option.Some(5);

            // Act
            var result = option.ValueOr(10);

            // Assert
            Assert.Equal(5, result);
        }

        [Fact]
        public void ValueOr_WithNone_ReturnsDefaultValue()
        {
            // Arrange
            var option = Option.None<int>();

            // Act
            var result = option.ValueOr(10);

            // Assert
            Assert.Equal(10, result);
        }

        [Fact]
        public void Match_WithSome_CallsSomeFunction()
        {
            // Arrange
            var option = Option.Some(5);
            bool someWasCalled = false;
            bool noneWasCalled = false;

            // Act
            option.Match(
                some: x => { someWasCalled = true; },
                none: () => { noneWasCalled = true; }
            );

            // Assert
            Assert.True(someWasCalled);
            Assert.False(noneWasCalled);
        }

        [Fact]
        public void Match_WithNone_CallsNoneFunction()
        {
            // Arrange
            var option = Option.None<int>();
            bool someWasCalled = false;
            bool noneWasCalled = false;

            // Act
            option.Match(
                some: x => { someWasCalled = true; },
                none: () => { noneWasCalled = true; }
            );

            // Assert
            Assert.False(someWasCalled);
            Assert.True(noneWasCalled);
        }

        [Fact]
        public void ToEnumerable_WithSome_ReturnsEnumerableWithOneElement()
        {
            // Arrange
            var option = Option.Some(5);

            // Act
            var result = option.ToEnumerable().ToList();

            // Assert
            Assert.Single(result);
            Assert.Equal(5, result[0]);
        }

        [Fact]
        public void ToEnumerable_WithNone_ReturnsEmptyEnumerable()
        {
            // Arrange
            var option = Option.None<int>();

            // Act
            var result = option.ToEnumerable().ToList();

            // Assert
            Assert.Empty(result);
        }

        [Fact]
        public void FromReference_WithNonNullValue_ReturnsSome()
        {
            // Arrange
            string value = "test";

            // Act
            var option = Option.FromReference(value);

            // Assert
            Assert.True(option.IsSome);
            Assert.Equal(value, option.Value);
        }

        [Fact]
        public void FromReference_WithNullValue_ReturnsNone()
        {
            // Arrange
            string? value = null;

            // Act
            var option = Option.FromReference<string>(value!);

            // Assert
            Assert.True(option.IsNone);
        }

        [Fact]
        public void Try_WithSuccessfulOperation_ReturnsSome()
        {
            // Act
            var option = Option.Try(() => int.Parse("123"));

            // Assert
            Assert.True(option.IsSome);
            Assert.Equal(123, option.Value);
        }

        [Fact]
        public void Try_WithFailingOperation_ReturnsNone()
        {
            // Act
            var option = Option.Try(() => int.Parse("not a number"));

            // Assert
            Assert.True(option.IsNone);
        }
    }
}
