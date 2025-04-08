using System;
using System.Collections.Generic;
using System.Linq;

namespace SemanticDuplicationDemo
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Semantic Duplication Demo");
            
            // Process user data with semantic duplication
            var users = GetUsers();
            ProcessUsers(users);
            
            // Process product data with semantic duplication
            var products = GetProducts();
            ProcessProducts(products);
            
            Console.WriteLine("Demo completed successfully!");
        }
        
        // Method 1 with semantic duplication
        static List<User> GetUsers()
        {
            Console.WriteLine("Getting users...");
            
            // Create sample data
            var data = new List<User>
            {
                new User { Id = 1, Name = "John Doe", Email = "john@example.com" },
                new User { Id = 2, Name = "Jane Smith", Email = "jane@example.com" },
                new User { Id = 3, Name = "Bob Johnson", Email = "bob@example.com" }
            };
            
            Console.WriteLine($"Retrieved {data.Count} users");
            return data;
        }
        
        // Method 2 with semantic duplication (similar to Method 1)
        static List<Product> GetProducts()
        {
            Console.WriteLine("Getting products...");
            
            // Create sample data
            var items = new List<Product>
            {
                new Product { Id = 101, Name = "Laptop", Price = 999.99m },
                new Product { Id = 102, Name = "Smartphone", Price = 499.99m },
                new Product { Id = 103, Name = "Headphones", Price = 99.99m }
            };
            
            Console.WriteLine($"Retrieved {items.Count} products");
            return items;
        }
        
        // Method 3 with semantic duplication
        static void ProcessUsers(List<User> users)
        {
            Console.WriteLine("Processing users...");
            
            // Validate data
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }
            
            if (!users.Any())
            {
                Console.WriteLine("No users to process");
                return;
            }
            
            // Process each item
            foreach (var user in users)
            {
                // Validate item
                if (string.IsNullOrEmpty(user.Name) || string.IsNullOrEmpty(user.Email))
                {
                    Console.WriteLine($"Invalid user data: {user.Id}");
                    continue;
                }
                
                // Process item
                Console.WriteLine($"Processing user: {user.Name}");
                
                // Simulate processing
                System.Threading.Thread.Sleep(100);
                
                // Log result
                Console.WriteLine($"User {user.Id} processed successfully");
            }
            
            Console.WriteLine("User processing completed");
        }
        
        // Method 4 with semantic duplication (similar to Method 3)
        static void ProcessProducts(List<Product> products)
        {
            Console.WriteLine("Processing products...");
            
            // Check input
            if (products is null)
            {
                throw new ArgumentNullException(nameof(products));
            }
            
            if (products.Count == 0)
            {
                Console.WriteLine("No products to process");
                return;
            }
            
            // Handle each product
            foreach (var item in products)
            {
                // Check validity
                if (string.IsNullOrEmpty(item.Name) || item.Price <= 0)
                {
                    Console.WriteLine($"Invalid product data: {item.Id}");
                    continue;
                }
                
                // Handle product
                Console.WriteLine($"Processing product: {item.Name}");
                
                // Wait a bit
                System.Threading.Thread.Sleep(100);
                
                // Output result
                Console.WriteLine($"Product {item.Id} processed successfully");
            }
            
            Console.WriteLine("Product processing completed");
        }
    }
    
    // Data models
    public class User
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Email { get; set; }
    }
    
    public class Product
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public decimal Price { get; set; }
    }
}
