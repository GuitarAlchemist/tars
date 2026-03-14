using System;
using ProductCatalog.Domain.Common;
using ProductCatalog.Domain.Entities;

namespace ProductCatalog.Domain.Events
{
    public class ProductCreatedEvent : DomainEvent
    {
        public Product Product { get; }
        
        public ProductCreatedEvent(Product product)
        {
            Product = product;
        }
    }
    
    public class ProductUpdatedEvent : DomainEvent
    {
        public Product Product { get; }
        
        public ProductUpdatedEvent(Product product)
        {
            Product = product;
        }
    }
    
    public class ProductStockUpdatedEvent : DomainEvent
    {
        public Product Product { get; }
        public int NewQuantity { get; }
        
        public ProductStockUpdatedEvent(Product product, int newQuantity)
        {
            Product = product;
            NewQuantity = newQuantity;
        }
    }
    
    public class ProductCategoryChangedEvent : DomainEvent
    {
        public Product Product { get; }
        
        public ProductCategoryChangedEvent(Product product)
        {
            Product = product;
        }
    }
    
    public class ProductPublishedEvent : DomainEvent
    {
        public Product Product { get; }
        
        public ProductPublishedEvent(Product product)
        {
            Product = product;
        }
    }
    
    public class ProductUnpublishedEvent : DomainEvent
    {
        public Product Product { get; }
        
        public ProductUnpublishedEvent(Product product)
        {
            Product = product;
        }
    }
    
    public class ProductReviewAddedEvent : DomainEvent
    {
        public Product Product { get; }
        public ProductReview Review { get; }
        
        public ProductReviewAddedEvent(Product product, ProductReview review)
        {
            Product = product;
            Review = review;
        }
    }
}
