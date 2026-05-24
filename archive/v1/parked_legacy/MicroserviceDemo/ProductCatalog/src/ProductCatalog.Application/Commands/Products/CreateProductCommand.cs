using System;
using System.Threading;
using System.Threading.Tasks;
using MediatR;
using ProductCatalog.Application.DTOs;
using ProductCatalog.Domain.Entities;
using ProductCatalog.Domain.Repositories;
using ProductCatalog.Domain.ValueObjects;

namespace ProductCatalog.Application.Commands.Products
{
    public class CreateProductCommand : IRequest<ProductDto>
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public decimal Price { get; set; }
        public string Currency { get; set; }
        public int StockQuantity { get; set; }
        public Guid CategoryId { get; set; }
    }
    
    public class CreateProductCommandHandler : IRequestHandler<CreateProductCommand, ProductDto>
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IMediator _mediator;
        
        public CreateProductCommandHandler(IUnitOfWork unitOfWork, IMediator mediator)
        {
            _unitOfWork = unitOfWork ?? throw new ArgumentNullException(nameof(unitOfWork));
            _mediator = mediator ?? throw new ArgumentNullException(nameof(mediator));
        }
        
        public async Task<ProductDto> Handle(CreateProductCommand request, CancellationToken cancellationToken)
        {
            // Get the category
            var category = await _unitOfWork.CategoryRepository.GetByIdAsync(request.CategoryId, cancellationToken);
            if (category == null)
            {
                throw new ApplicationException($"Category with ID {request.CategoryId} not found");
            }
            
            // Create money value object
            var price = new Money(request.Price, request.Currency);
            
            // Create the product
            var product = new Product(
                request.Name,
                request.Description,
                price,
                request.StockQuantity,
                category);
            
            // Save to repository
            await _unitOfWork.ProductRepository.AddAsync(product, cancellationToken);
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
                Reviews = new List<ProductReviewDto>()
            };
        }
    }
}
