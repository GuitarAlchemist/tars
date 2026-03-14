using ProductCatalog.Domain.Common;
using ProductCatalog.Domain.Entities;

namespace ProductCatalog.Domain.Events
{
    public class CategoryCreatedEvent : DomainEvent
    {
        public ProductCategory Category { get; }
        
        public CategoryCreatedEvent(ProductCategory category)
        {
            Category = category;
        }
    }
    
    public class CategoryUpdatedEvent : DomainEvent
    {
        public ProductCategory Category { get; }
        
        public CategoryUpdatedEvent(ProductCategory category)
        {
            Category = category;
        }
    }
    
    public class CategoryParentChangedEvent : DomainEvent
    {
        public ProductCategory Category { get; }
        
        public CategoryParentChangedEvent(ProductCategory category)
        {
            Category = category;
        }
    }
    
    public class CategorySubcategoryAddedEvent : DomainEvent
    {
        public ProductCategory ParentCategory { get; }
        public ProductCategory Subcategory { get; }
        
        public CategorySubcategoryAddedEvent(ProductCategory parentCategory, ProductCategory subcategory)
        {
            ParentCategory = parentCategory;
            Subcategory = subcategory;
        }
    }
}
