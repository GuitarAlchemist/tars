using System;
using System.Linq;
using Xunit;
using TarsEngine.FSharp.Core.TreeOfThought;

namespace TarsEngine.FSharp.Tests.TreeOfThought
{
    public class ThoughtNodeTests
    {
        [Fact]
        public void CreateNode_ShouldCreateNodeWithCorrectThought()
        {
            // Arrange
            var thought = "Test thought";

            // Act
            var node = ThoughtNode.createNode(thought);

            // Assert
            Assert.Equal(thought, node.Thought);
            Assert.Empty(node.Children);
            Assert.True(Microsoft.FSharp.Core.FSharpOption<ThoughtNode.EvaluationMetrics>.get_IsNone(node.Evaluation));
            Assert.False(node.Pruned);
            Assert.Empty(node.Metadata);
        }

        [Fact]
        public void AddChild_ShouldAddChildToNode()
        {
            // Arrange
            var parent = ThoughtNode.createNode("Parent");
            var child = ThoughtNode.createNode("Child");

            // Act
            var updatedParent = ThoughtNode.addChild(parent, child);

            // Assert
            Assert.Single(updatedParent.Children);
            Assert.Equal("Child", updatedParent.Children[0].Thought);
        }

        [Fact]
        public void EvaluateNode_ShouldSetEvaluationMetrics()
        {
            // Arrange
            var node = ThoughtNode.createNode("Test");
            var metrics = ThoughtNode.createMetrics(0.8, 0.7, 0.6, 0.5);

            // Act
            var evaluatedNode = ThoughtNode.evaluateNode(node, metrics);

            // Assert
            Assert.False(Microsoft.FSharp.Core.FSharpOption<ThoughtNode.EvaluationMetrics>.get_IsNone(evaluatedNode.Evaluation));
            var actualMetrics = ((Microsoft.FSharp.Core.FSharpOption<ThoughtNode.EvaluationMetrics>)evaluatedNode.Evaluation).Value;
            Assert.Equal(0.8, actualMetrics.Correctness);
            Assert.Equal(0.7, actualMetrics.Efficiency);
            Assert.Equal(0.6, actualMetrics.Robustness);
            Assert.Equal(0.5, actualMetrics.Maintainability);
            Assert.Equal(0.65, actualMetrics.Overall);
        }

        [Fact]
        public void PruneNode_ShouldMarkNodeAsPruned()
        {
            // Arrange
            var node = ThoughtNode.createNode("Test");

            // Act
            var prunedNode = ThoughtNode.pruneNode(node);

            // Assert
            Assert.True(prunedNode.Pruned);
        }

        [Fact]
        public void AddMetadata_ShouldAddMetadataToNode()
        {
            // Arrange
            var node = ThoughtNode.createNode("Test");
            var key = "key";
            var value = "value";

            // Act
            var updatedNode = ThoughtNode.addMetadata(node, key, value);

            // Assert
            Assert.Single(updatedNode.Metadata);
            Assert.Equal(value, updatedNode.Metadata[key]);
        }

        [Fact]
        public void GetMetadata_ShouldReturnMetadataValue()
        {
            // Arrange
            var node = ThoughtNode.createNode("Test");
            var key = "key";
            var value = "value";
            var updatedNode = ThoughtNode.addMetadata(node, key, value);

            // Act
            var option = ThoughtNode.getMetadata<string>(updatedNode, key);

            // Assert
            Assert.False(Microsoft.FSharp.Core.FSharpOption<string>.get_IsNone(option));
            Assert.Equal(value, ((Microsoft.FSharp.Core.FSharpOption<string>)option).Value);
        }

        [Fact]
        public void GetScore_ShouldReturnOverallScore()
        {
            // Arrange
            var node = ThoughtNode.createNode("Test");
            var metrics = ThoughtNode.createMetrics(0.8, 0.7, 0.6, 0.5);
            var evaluatedNode = ThoughtNode.evaluateNode(node, metrics);

            // Act
            var score = ThoughtNode.getScore(evaluatedNode);

            // Assert
            Assert.Equal(0.65, score);
        }
    }
}
