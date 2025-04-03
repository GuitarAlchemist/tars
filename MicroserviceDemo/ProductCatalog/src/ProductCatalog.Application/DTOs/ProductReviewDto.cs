using System;

namespace ProductCatalog.Application.DTOs
{
    public class ProductReviewDto
    {
        public Guid Id { get; set; }
        public string Title { get; set; }
        public string Content { get; set; }
        public int Rating { get; set; }
        public string ReviewerName { get; set; }
        public DateTime CreatedDate { get; set; }
    }
}
