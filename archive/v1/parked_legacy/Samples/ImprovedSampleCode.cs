using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Samples
{
    public class ImprovedSampleCode
    {
        // Constants for better maintainability
        private const decimal TaxRate = 0.08m;
        private const decimal HighValueOrderThreshold = 100m;
        private const decimal MediumValueOrderThreshold = 50m;
        private const decimal HighValueDiscount = 5m;
        private const decimal MediumValueDiscount = 2m;
        
        // Status constants for better readability
        private const int StatusPending = 1;
        
        // This method has been improved for performance
        public void ProcessData(List<int> data)
        {
            // Validate input
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }
            
            // Process each threshold value only once
            for (int i = 0; i < data.Count; i++)
            {
                // Move LINQ operation outside the inner loop
                var filteredData = data.Where(x => x > i).ToList();
                
                foreach (var item in filteredData)
                {
                    Console.WriteLine($"Processing item: {item}");
                    
                    // Use StringBuilder instead of string concatenation
                    var resultBuilder = new StringBuilder();
                    for (int j = 0; j < item; j++)
                    {
                        resultBuilder.Append(j);
                    }
                    
                    Console.WriteLine(resultBuilder.ToString());
                }
            }
        }
        
        // This method has been improved with proper error handling
        public int CalculateAverage(List<int> numbers)
        {
            // Add null check
            if (numbers == null)
            {
                throw new ArgumentNullException(nameof(numbers));
            }
            
            // Add empty list check
            if (numbers.Count == 0)
            {
                throw new ArgumentException("Cannot calculate average of an empty list", nameof(numbers));
            }
            
            int sum = 0;
            foreach (var number in numbers)
            {
                sum += number;
            }
            
            // No risk of division by zero now
            return sum / numbers.Count;
        }
        
        // This method has been improved for maintainability
        public void ProcessOrder(Order order)
        {
            // Validate input
            if (order == null)
            {
                throw new ArgumentNullException(nameof(order));
            }
            
            // Use named constants for better readability
            if (order.Status == StatusPending)
            {
                // Apply tax using named constant
                order.Total = order.Subtotal * (1 + TaxRate);
                
                // Apply discount based on order value
                ApplyDiscountBasedOnValue(order);
                
                // Use extracted method to avoid code duplication
                LogOrderDetails(order);
            }
        }
        
        // Extracted method for applying discounts
        private void ApplyDiscountBasedOnValue(Order order)
        {
            if (order.Total > HighValueOrderThreshold)
            {
                order.ApplyDiscount(HighValueDiscount);
            }
            else if (order.Total > MediumValueOrderThreshold)
            {
                order.ApplyDiscount(MediumValueDiscount);
            }
        }
        
        // Extracted method for logging order details
        private void LogOrderDetails(Order order)
        {
            Console.WriteLine($"Order processed: {order.Id}");
            Console.WriteLine($"Order total: {order.Total}");
            Console.WriteLine($"Order date: {order.Date}");
        }
    }
    
    // No changes needed to the Order class
    public class Order
    {
        public int Id { get; set; }
        public decimal Subtotal { get; set; }
        public decimal Total { get; set; }
        public int Status { get; set; }
        public DateTime Date { get; set; }
        
        public void ApplyDiscount(decimal percentage)
        {
            Total -= Total * (percentage / 100);
        }
    }
}
