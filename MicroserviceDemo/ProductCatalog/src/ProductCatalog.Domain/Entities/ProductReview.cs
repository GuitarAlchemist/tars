using System;
using ProductCatalog.Domain.Common;

namespace ProductCatalog.Domain.Entities
{
    public class ProductReview : Entity
    {
        // Private constructor for EF Core
        private ProductReview() { }
        
        public ProductReview(string title, string content, int rating, string reviewerName)
        {
            Id = Guid.NewGuid();
            Title = title ?? throw new ArgumentNullException(nameof(title));
            Content = content ?? throw new ArgumentNullException(nameof(content));
            
            if (rating < 1 || rating > 5)
                throw new ArgumentOutOfRangeException(nameof(rating), "Rating must be between 1 and 5");
                
            Rating = rating;
            ReviewerName = reviewerName ?? throw new ArgumentNullException(nameof(reviewerName));
            CreatedDate = DateTime.UtcNow;
        }
        
        public string Title { get; private set; }
        public string Content { get; private set; }
        public int Rating { get; private set; }
        public string ReviewerName { get; private set; }
        public DateTime CreatedDate { get; private set; }
        
        public void UpdateReview(string title, string content, int rating)
        {
            Title = title ?? throw new ArgumentNullException(nameof(title));
            Content = content ?? throw new ArgumentNullException(nameof(content));
            
            if (rating < 1 || rating > 5)
                throw new ArgumentOutOfRangeException(nameof(rating), "Rating must be between 1 and 5");
                
            Rating = rating;
        }
    }
}
