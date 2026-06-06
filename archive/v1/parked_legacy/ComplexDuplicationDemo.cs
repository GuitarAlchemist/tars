using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ComplexDuplicationDemo
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Complex Duplication Demo");
            
            // Process user data
            var userData = await GetUserDataAsync();
            ProcessUserData(userData);
            
            // Process product data
            var productData = await GetProductDataAsync();
            ProcessProductData(productData);
            
            // Process order data
            var orderData = await GetOrderDataAsync();
            ProcessOrderData(orderData);
            
            Console.WriteLine("Demo completed successfully!");
        }
        
        // Duplicated method pattern 1
        static async Task<List<User>> GetUserDataAsync()
        {
            Console.WriteLine("Fetching user data...");
            
            // Simulate API call
            await Task.Delay(500);
            
            // Create sample data
            var users = new List<User>
            {
                new User { Id = 1, Name = "John Doe", Email = "john@example.com" },
                new User { Id = 2, Name = "Jane Smith", Email = "jane@example.com" },
                new User { Id = 3, Name = "Bob Johnson", Email = "bob@example.com" }
            };
            
            Console.WriteLine($"Retrieved {users.Count} users");
            return users;
        }
        
        // Duplicated method pattern 2
        static async Task<List<Product>> GetProductDataAsync()
        {
            Console.WriteLine("Fetching product data...");
            
            // Simulate API call
            await Task.Delay(500);
            
            // Create sample data
            var products = new List<Product>
            {
                new Product { Id = 101, Name = "Laptop", Price = 999.99m },
                new Product { Id = 102, Name = "Smartphone", Price = 499.99m },
                new Product { Id = 103, Name = "Headphones", Price = 99.99m }
            };
            
            Console.WriteLine($"Retrieved {products.Count} products");
            return products;
        }
        
        // Duplicated method pattern 3
        static async Task<List<Order>> GetOrderDataAsync()
        {
            Console.WriteLine("Fetching order data...");
            
            // Simulate API call
            await Task.Delay(500);
            
            // Create sample data
            var orders = new List<Order>
            {
                new Order { Id = 1001, UserId = 1, ProductId = 101, Quantity = 1 },
                new Order { Id = 1002, UserId = 2, ProductId = 102, Quantity = 2 },
                new Order { Id = 1003, UserId = 3, ProductId = 103, Quantity = 3 }
            };
            
            Console.WriteLine($"Retrieved {orders.Count} orders");
            return orders;
        }
        
        // Duplicated processing pattern 1
        static void ProcessUserData(List<User> users)
        {
            Console.WriteLine("Processing user data...");
            
            // Validate data
            if (users == null || !users.Any())
            {
                Console.WriteLine("No user data to process");
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
            
            Console.WriteLine("User data processing completed");
        }
        
        // Duplicated processing pattern 2
        static void ProcessProductData(List<Product> products)
        {
            Console.WriteLine("Processing product data...");
            
            // Validate data
            if (products == null || !products.Any())
            {
                Console.WriteLine("No product data to process");
                return;
            }
            
            // Process each item
            foreach (var product in products)
            {
                // Validate item
                if (string.IsNullOrEmpty(product.Name) || product.Price <= 0)
                {
                    Console.WriteLine($"Invalid product data: {product.Id}");
                    continue;
                }
                
                // Process item
                Console.WriteLine($"Processing product: {product.Name}");
                
                // Simulate processing
                System.Threading.Thread.Sleep(100);
                
                // Log result
                Console.WriteLine($"Product {product.Id} processed successfully");
            }
            
            Console.WriteLine("Product data processing completed");
        }
        
        // Duplicated processing pattern 3
        static void ProcessOrderData(List<Order> orders)
        {
            Console.WriteLine("Processing order data...");
            
            // Validate data
            if (orders == null || !orders.Any())
            {
                Console.WriteLine("No order data to process");
                return;
            }
            
            // Process each item
            foreach (var order in orders)
            {
                // Validate item
                if (order.UserId <= 0 || order.ProductId <= 0 || order.Quantity <= 0)
                {
                    Console.WriteLine($"Invalid order data: {order.Id}");
                    continue;
                }
                
                // Process item
                Console.WriteLine($"Processing order: {order.Id}");
                
                // Simulate processing
                System.Threading.Thread.Sleep(100);
                
                // Log result
                Console.WriteLine($"Order {order.Id} processed successfully");
            }
            
            Console.WriteLine("Order data processing completed");
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
    
    public class Order
    {
        public int Id { get; set; }
        public int UserId { get; set; }
        public int ProductId { get; set; }
        public int Quantity { get; set; }
    }
}
