using System;
using System.Threading;
using System.Threading.Tasks;
using MediatR;
using ProductCatalog.Application.DTOs;
using ProductCatalog.Domain.Repositories;
using ProductCatalog.Domain.ValueObjects;

namespace ProductCatalog.Application.Commands.Products
{
    public class UpdateProductCommand : IRequest<ProductDto>
    {
        public Guid Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public decimal Price { get; set; }
        public string Currency { get; set; }
        public int StockQuantity { get; set; }
        public Guid CategoryId { get; set; }
    }
    
    public class UpdateProductCommandHandler : IRequestHandler<UpdateProductCommand, ProductDto>
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IMediator _mediator;
        
        public UpdateProductCommandHandler(IUnitOfWork unitOfWork, IMediator mediator)
        {
            _unitOfWork = unitOfWork ?? throw new ArgumentNullException(nameof(unitOfWork));
            _mediator = mediator ?? throw new ArgumentNullException(nameof(mediator));
        }
        
        public async Task<ProductDto> Handle(UpdateProductCommand request, CancellationToken cancellationToken)
        {
            // Get the product
            var product = await _unitOfWork.ProductRepository.GetByIdAsync(request.Id, cancellationToken);
            if (product == null)
            {
                throw new ApplicationException($"Product with ID {request.Id} not found");
            }
            
            // Get the category if it's changing
            if (product.Category.Id != request.CategoryId)
            {
                var category = await _unitOfWork.CategoryRepository.GetByIdAsync(request.CategoryId, cancellationToken);
                if (category == null)
                {
                    throw new ApplicationException($"Category with ID {request.CategoryId} not found");
                }
                
                product.ChangeCategory(category);
            }
            
            // Create money value object
            var price = new Money(request.Price, request.Currency);
            
            // Update the product
            product.UpdateDetails(request.Name, request.Description, price);
            product.UpdateStock(request.StockQuantity);
            
            // Save to repository
            await _unitOfWork.ProductRepository.UpdateAsync(product, cancellationToken);
            await _unitOfWork.SaveEntitiesAsync(cancellationToken);
            
            // Return DTO
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
