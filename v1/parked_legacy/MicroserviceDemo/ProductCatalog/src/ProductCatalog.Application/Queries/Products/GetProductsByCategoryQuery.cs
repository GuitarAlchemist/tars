using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MediatR;
using ProductCatalog.Application.DTOs;
using ProductCatalog.Domain.Repositories;

namespace ProductCatalog.Application.Queries.Products
{
    public class GetProductsByCategoryQuery : IRequest<IEnumerable<ProductDto>>
    {
        public Guid CategoryId { get; set; }
    }
    
    public class GetProductsByCategoryQueryHandler : IRequestHandler<GetProductsByCategoryQuery, IEnumerable<ProductDto>>
    {
        private readonly IUnitOfWork _unitOfWork;
        
        public GetProductsByCategoryQueryHandler(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork ?? throw new ArgumentNullException(nameof(unitOfWork));
        }
        
        public async Task<IEnumerable<ProductDto>> Handle(GetProductsByCategoryQuery request, CancellationToken cancellationToken)
        {
            var products = await _unitOfWork.ProductRepository.GetProductsByCategoryAsync(request.CategoryId, cancellationToken);
            
            return products.Select(p => new ProductDto
            {
                Id = p.Id,
                Name = p.Name,
                Description = p.Description,
                Price = p.Price.Amount,
                Currency = p.Price.Currency,
                StockQuantity = p.StockQuantity,
                CategoryId = p.Category.Id,
                CategoryName = p.Category.Name,
                Status = p.Status.ToString(),
                CreatedDate = p.CreatedDate,
                LastModifiedDate = p.LastModifiedDate,
                Reviews = p.Reviews.Select(r => new ProductReviewDto
                {
                    Id = r.Id,
                    Title = r.Title,
                    Content = r.Content,
                    Rating = r.Rating,
                    ReviewerName = r.ReviewerName,
                    CreatedDate = r.CreatedDate
                })
            });
        }
    }
}
