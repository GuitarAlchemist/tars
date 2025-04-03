using System;
using System.Collections.Generic;

namespace ProductCatalog.Application.DTOs
{
    public class CategoryDto
    {
        public Guid Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public Guid? ParentId { get; set; }
        public string ParentName { get; set; }
        public DateTime CreatedDate { get; set; }
        public DateTime LastModifiedDate { get; set; }
        public IEnumerable<CategoryDto> Subcategories { get; set; }
    }
}
