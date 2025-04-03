using System;
using System.Collections.Generic;
using ProductCatalog.Domain.Common;
using ProductCatalog.Domain.ValueObjects;
using ProductCatalog.Domain.Events;

namespace ProductCatalog.Domain.Entities
{
    public class Product : Entity, IAggregateRoot
    {
        private readonly List<ProductReview> _reviews = new List<ProductReview>();
        
        // Private constructor for EF Core
        private Product() { }
        
        public Product(
            string name, 
            string description, 
            Money price, 
            int stockQuantity, 
            ProductCategory category)
        {
            Id = Guid.NewGuid();
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Description = description;
            Price = price ?? throw new ArgumentNullException(nameof(price));
            StockQuantity = stockQuantity;
            Category = category ?? throw new ArgumentNullException(nameof(category));
            Status = ProductStatus.Draft;
            CreatedDate = DateTime.UtcNow;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductCreatedEvent(this));
        }
        
        public string Name { get; private set; }
        public string Description { get; private set; }
        public Money Price { get; private set; }
        public int StockQuantity { get; private set; }
        public ProductCategory Category { get; private set; }
        public ProductStatus Status { get; private set; }
        public DateTime CreatedDate { get; private set; }
        public DateTime LastModifiedDate { get; private set; }
        public IReadOnlyCollection<ProductReview> Reviews => _reviews.AsReadOnly();
        
        public void UpdateDetails(string name, string description, Money price)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Description = description;
            Price = price ?? throw new ArgumentNullException(nameof(price));
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductUpdatedEvent(this));
        }
        
        public void UpdateStock(int quantity)
        {
            if (quantity < 0)
                throw new ArgumentException("Quantity cannot be negative", nameof(quantity));
                
            StockQuantity = quantity;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductStockUpdatedEvent(this, quantity));
        }
        
        public void ChangeCategory(ProductCategory category)
        {
            Category = category ?? throw new ArgumentNullException(nameof(category));
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductCategoryChangedEvent(this));
        }
        
        public void Publish()
        {
            if (Status == ProductStatus.Published)
                return;
                
            Status = ProductStatus.Published;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductPublishedEvent(this));
        }
        
        public void Unpublish()
        {
            if (Status == ProductStatus.Draft)
                return;
                
            Status = ProductStatus.Draft;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductUnpublishedEvent(this));
        }
        
        public void AddReview(ProductReview review)
        {
            if (review == null)
                throw new ArgumentNullException(nameof(review));
                
            _reviews.Add(review);
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new ProductReviewAddedEvent(this, review));
        }
    }
}
