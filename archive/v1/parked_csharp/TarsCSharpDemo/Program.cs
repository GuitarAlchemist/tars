using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

// TARS C# Programming Learning Demonstration (Simplified)
namespace TarsLearning
{
    public class TarsProgrammingLearner
    {
        private readonly List<string> _masteredConcepts = new();

        public async Task<string> LearnAsync(string concept)
        {
            Console.WriteLine($"🧠 TARS Learning: {concept}");
            await Task.Delay(50);

            if (!_masteredConcepts.Contains(concept))
            {
                _masteredConcepts.Add(concept);
                Console.WriteLine($"✅ Mastered: {concept}");
            }
            return concept;
        }

        public void DisplaySkills()
        {
            Console.WriteLine($"\n📊 TARS has mastered {_masteredConcepts.Count} concepts:");
            foreach (var concept in _masteredConcepts)
            {
                Console.WriteLine($"   ✅ {concept}");
            }
        }
    }

    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 TARS C# Programming Learning Demo");
            Console.WriteLine("====================================");

            var learner = new TarsProgrammingLearner();

            var concepts = new[] { "LINQ", "Async/Await", "Generics", "Pattern Matching" };
            foreach (var concept in concepts)
            {
                await learner.LearnAsync(concept);
            }

            learner.DisplaySkills();

            // Demonstrate LINQ
            Console.WriteLine("\n🔧 LINQ Demonstration:");
            var numbers = Enumerable.Range(1, 10);
            var result = numbers
                .Where(x => x % 2 == 0)
                .Select(x => x * x)
                .Sum();
            Console.WriteLine($"Sum of squares of even numbers 1-10: {result}");

            Console.WriteLine("\n🎉 TARS C# Learning: SUCCESSFUL!");
        }
    }
}
