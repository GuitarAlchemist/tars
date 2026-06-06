using System;
using System.Linq;
using Xunit;
using TarsEngine.FSharp.Core.TreeOfThought;

namespace TarsEngine.FSharp.Tests.TreeOfThought
{
    public class ThoughtTreeTests
    {
        [Fact]
        public void Depth_ShouldReturnCorrectDepth()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child = ThoughtNode.createNode("Child");
            var grandchild = ThoughtNode.createNode("Grandchild");
            
            var childWithGrandchild = ThoughtNode.addChild(child, grandchild);
            var rootWithChild = ThoughtNode.addChild(root, childWithGrandchild);

            // Act
            var depth = ThoughtTree.depth(rootWithChild);

            // Assert
            Assert.Equal(3, depth);
        }

        [Fact]
        public void BreadthAtLevel_ShouldReturnCorrectBreadth()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child1 = ThoughtNode.createNode("Child1");
            var child2 = ThoughtNode.createNode("Child2");
            
            var rootWithChildren = ThoughtNode.addChild(ThoughtNode.addChild(root, child1), child2);

            // Act
            var breadth = ThoughtTree.breadthAtLevel(1, rootWithChildren);

            // Assert
            Assert.Equal(2, breadth);
        }

        [Fact]
        public void MaxBreadth_ShouldReturnMaximumBreadth()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child1 = ThoughtNode.createNode("Child1");
            var child2 = ThoughtNode.createNode("Child2");
            var grandchild1 = ThoughtNode.createNode("Grandchild1");
            var grandchild2 = ThoughtNode.createNode("Grandchild2");
            var grandchild3 = ThoughtNode.createNode("Grandchild3");
            
            var child1WithGrandchildren = ThoughtNode.addChild(ThoughtNode.addChild(child1, grandchild1), grandchild2);
            var child2WithGrandchild = ThoughtNode.addChild(child2, grandchild3);
            var rootWithChildren = ThoughtNode.addChild(ThoughtNode.addChild(root, child1WithGrandchildren), child2WithGrandchild);

            // Act
            var maxBreadth = ThoughtTree.maxBreadth(rootWithChildren);

            // Assert
            Assert.Equal(3, maxBreadth);
        }

        [Fact]
        public void FindNode_ShouldReturnNodeWithMatchingThought()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child1 = ThoughtNode.createNode("Child1");
            var child2 = ThoughtNode.createNode("Child2");
            
            var rootWithChildren = ThoughtNode.addChild(ThoughtNode.addChild(root, child1), child2);

            // Act
            var option = ThoughtTree.findNode("Child2", rootWithChildren);

            // Assert
            Assert.False(Microsoft.FSharp.Core.FSharpOption<ThoughtNode.ThoughtNode>.get_IsNone(option));
            Assert.Equal("Child2", ((Microsoft.FSharp.Core.FSharpOption<ThoughtNode.ThoughtNode>)option).Value.Thought);
        }

        [Fact]
        public void SelectBestNode_ShouldReturnNodeWithHighestScore()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child1 = ThoughtNode.createNode("Child1");
            var child2 = ThoughtNode.createNode("Child2");
            
            var metrics1 = ThoughtNode.createMetrics(0.6, 0.6, 0.6, 0.6);
            var metrics2 = ThoughtNode.createMetrics(0.8, 0.8, 0.8, 0.8);
            
            var evaluatedChild1 = ThoughtNode.evaluateNode(child1, metrics1);
            var evaluatedChild2 = ThoughtNode.evaluateNode(child2, metrics2);
            
            var rootWithChildren = ThoughtNode.addChild(ThoughtNode.addChild(root, evaluatedChild1), evaluatedChild2);

            // Act
            var bestNode = ThoughtTree.selectBestNode(rootWithChildren);

            // Assert
            Assert.Equal("Child2", bestNode.Thought);
        }

        [Fact]
        public void PruneByThreshold_ShouldPruneNodesBelowThreshold()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child1 = ThoughtNode.createNode("Child1");
            var child2 = ThoughtNode.createNode("Child2");
            
            var metrics1 = ThoughtNode.createMetrics(0.6, 0.6, 0.6, 0.6);
            var metrics2 = ThoughtNode.createMetrics(0.8, 0.8, 0.8, 0.8);
            
            var evaluatedChild1 = ThoughtNode.evaluateNode(child1, metrics1);
            var evaluatedChild2 = ThoughtNode.evaluateNode(child2, metrics2);
            
            var rootWithChildren = ThoughtNode.addChild(ThoughtNode.addChild(root, evaluatedChild1), evaluatedChild2);

            // Act
            var prunedTree = ThoughtTree.pruneByThreshold(0.7, rootWithChildren);

            // Assert
            Assert.Equal(2, prunedTree.Children.Length);
            Assert.True(prunedTree.Children[0].Pruned);
            Assert.False(prunedTree.Children[1].Pruned);
        }

        [Fact]
        public void CountNodes_ShouldReturnTotalNumberOfNodes()
        {
            // Arrange
            var root = ThoughtNode.createNode("Root");
            var child1 = ThoughtNode.createNode("Child1");
            var child2 = ThoughtNode.createNode("Child2");
            var grandchild = ThoughtNode.createNode("Grandchild");
            
            var child1WithGrandchild = ThoughtNode.addChild(child1, grandchild);
            var rootWithChildren = ThoughtNode.addChild(ThoughtNode.addChild(root, child1WithGrandchild), child2);

            // Act
            var count = ThoughtTree.countNodes(rootWithChildren);

            // Assert
            Assert.Equal(4, count);
        }
    }
}
