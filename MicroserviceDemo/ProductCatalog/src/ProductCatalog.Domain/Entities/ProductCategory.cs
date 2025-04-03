using System;
using System.Collections.Generic;
using ProductCatalog.Domain.Common;
using ProductCatalog.Domain.Events;

namespace ProductCatalog.Domain.Entities
{
    public class ProductCategory : Entity, IAggregateRoot
    {
        private readonly List<ProductCategory> _subcategories = new List<ProductCategory>();
        
        // Private constructor for EF Core
        private ProductCategory() { }
        
        public ProductCategory(string name, string description = null, ProductCategory parent = null)
        {
            Id = Guid.NewGuid();
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Description = description;
            Parent = parent;
            CreatedDate = DateTime.UtcNow;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new CategoryCreatedEvent(this));
        }
        
        public string Name { get; private set; }
        public string Description { get; private set; }
        public ProductCategory Parent { get; private set; }
        public DateTime CreatedDate { get; private set; }
        public DateTime LastModifiedDate { get; private set; }
        public IReadOnlyCollection<ProductCategory> Subcategories => _subcategories.AsReadOnly();
        
        public void UpdateDetails(string name, string description)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Description = description;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new CategoryUpdatedEvent(this));
        }
        
        public void ChangeParent(ProductCategory parent)
        {
            // Prevent circular references
            if (parent != null && IsAncestorOf(parent))
                throw new InvalidOperationException("Cannot set a subcategory as parent (circular reference)");
                
            Parent = parent;
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new CategoryParentChangedEvent(this));
        }
        
        public void AddSubcategory(ProductCategory subcategory)
        {
            if (subcategory == null)
                throw new ArgumentNullException(nameof(subcategory));
                
            // Prevent circular references
            if (subcategory.IsAncestorOf(this))
                throw new InvalidOperationException("Cannot add an ancestor as subcategory (circular reference)");
                
            _subcategories.Add(subcategory);
            subcategory.ChangeParent(this);
            LastModifiedDate = DateTime.UtcNow;
            
            AddDomainEvent(new CategorySubcategoryAddedEvent(this, subcategory));
        }
        
        public bool IsAncestorOf(ProductCategory category)
        {
            if (category == null)
                return false;
                
            var current = category.Parent;
            
            while (current != null)
            {
                if (current.Id == Id)
                    return true;
                    
                current = current.Parent;
            }
            
            return false;
        }
    }
}
