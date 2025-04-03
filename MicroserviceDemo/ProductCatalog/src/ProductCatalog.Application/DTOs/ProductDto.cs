using System;
using System.Collections.Generic;

namespace ProductCatalog.Application.DTOs
{
    public class ProductDto
    {
        public Guid Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public decimal Price { get; set; }
        public string Currency { get; set; }
        public int StockQuantity { get; set; }
        public Guid CategoryId { get; set; }
        public string CategoryName { get; set; }
        public string Status { get; set; }
        public DateTime CreatedDate { get; set; }
        public DateTime LastModifiedDate { get; set; }
        public IEnumerable<ProductReviewDto> Reviews { get; set; }
    }
}
