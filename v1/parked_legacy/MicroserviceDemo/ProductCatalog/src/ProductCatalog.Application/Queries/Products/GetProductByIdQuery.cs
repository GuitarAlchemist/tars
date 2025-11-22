using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MediatR;
using ProductCatalog.Application.DTOs;
using ProductCatalog.Domain.Repositories;

namespace ProductCatalog.Application.Queries.Products
{
    public class GetProductByIdQuery : IRequest<ProductDto>
    {
        public Guid Id { get; set; }
    }
    
    public class GetProductByIdQueryHandler : IRequestHandler<GetProductByIdQuery, ProductDto>
    {
        private readonly IUnitOfWork _unitOfWork;
        
        public GetProductByIdQueryHandler(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork ?? throw new ArgumentNullException(nameof(unitOfWork));
        }
        
        public async Task<ProductDto> Handle(GetProductByIdQuery request, CancellationToken cancellationToken)
        {
            var product = await _unitOfWork.ProductRepository.GetByIdAsync(request.Id, cancellationToken);
            
            if (product == null)
            {
                return null;
            }
            
            return new ProductDto
            {
                Id = product.Id,
                Name = product.Name,
                Description = product.Description,
                Price = product.Price.Amount,
                Currency = product.Price.Currency,
                StockQuantity = product.StockQuantity,
                CategoryId = product.Category.Id,
                CategoryName = product.Category.Name,
                Status = product.Status.ToString(),
                CreatedDate = product.CreatedDate,
                LastModifiedDate = product.LastModifiedDate,
                Reviews = product.Reviews.Select(r => new ProductReviewDto
                {
                    Id = r.Id,
                    Title = r.Title,
                    Content = r.Content,
                    Rating = r.Rating,
                    ReviewerName = r.ReviewerName,
                    CreatedDate = r.CreatedDate
                })
            };
        }
    }
}
