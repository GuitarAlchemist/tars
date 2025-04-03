using System;
using System.Collections.Generic;
using System.Net;
using System.Threading.Tasks;
using MediatR;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using ProductCatalog.Application.Commands.Products;
using ProductCatalog.Application.DTOs;
using ProductCatalog.Application.Queries.Products;

namespace ProductCatalog.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ProductsController : ControllerBase
    {
        private readonly IMediator _mediator;
        private readonly ILogger<ProductsController> _logger;
        
        public ProductsController(IMediator mediator, ILogger<ProductsController> logger)
        {
            _mediator = mediator ?? throw new ArgumentNullException(nameof(mediator));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        [HttpGet]
        [ProducesResponseType(typeof(IEnumerable<ProductDto>), (int)HttpStatusCode.OK)]
        public async Task<ActionResult<IEnumerable<ProductDto>>> GetProducts([FromQuery] string searchTerm)
        {
            if (!string.IsNullOrEmpty(searchTerm))
            {
                var query = new SearchProductsQuery { SearchTerm = searchTerm };
                var products = await _mediator.Send(query);
                return Ok(products);
            }
            else
            {
                var query = new GetAllProductsQuery();
                var products = await _mediator.Send(query);
                return Ok(products);
            }
        }
        
        [HttpGet("{id}", Name = "GetProduct")]
        [ProducesResponseType(typeof(ProductDto), (int)HttpStatusCode.OK)]
        [ProducesResponseType((int)HttpStatusCode.NotFound)]
        public async Task<ActionResult<ProductDto>> GetProduct(Guid id)
        {
            var query = new GetProductByIdQuery { Id = id };
            var product = await _mediator.Send(query);
            
            if (product == null)
            {
                return NotFound();
            }
            
            return Ok(product);
        }
        
        [HttpGet("category/{categoryId}")]
        [ProducesResponseType(typeof(IEnumerable<ProductDto>), (int)HttpStatusCode.OK)]
        public async Task<ActionResult<IEnumerable<ProductDto>>> GetProductsByCategory(Guid categoryId)
        {
            var query = new GetProductsByCategoryQuery { CategoryId = categoryId };
            var products = await _mediator.Send(query);
            return Ok(products);
        }
        
        [HttpPost]
        [ProducesResponseType(typeof(ProductDto), (int)HttpStatusCode.Created)]
        [ProducesResponseType((int)HttpStatusCode.BadRequest)]
        public async Task<ActionResult<ProductDto>> CreateProduct([FromBody] CreateProductCommand command)
        {
            try
            {
                var product = await _mediator.Send(command);
                return CreatedAtRoute("GetProduct", new { id = product.Id }, product);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating product");
                return BadRequest(ex.Message);
            }
        }
        
        [HttpPut("{id}")]
        [ProducesResponseType((int)HttpStatusCode.NoContent)]
        [ProducesResponseType((int)HttpStatusCode.BadRequest)]
        [ProducesResponseType((int)HttpStatusCode.NotFound)]
        public async Task<IActionResult> UpdateProduct(Guid id, [FromBody] UpdateProductCommand command)
        {
            if (id != command.Id)
            {
                return BadRequest("ID in URL does not match ID in request body");
            }
            
            try
            {
                var product = await _mediator.Send(command);
                
                if (product == null)
                {
                    return NotFound();
                }
                
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating product");
                return BadRequest(ex.Message);
            }
        }
        
        [HttpDelete("{id}")]
        [ProducesResponseType((int)HttpStatusCode.NoContent)]
        [ProducesResponseType((int)HttpStatusCode.NotFound)]
        public async Task<IActionResult> DeleteProduct(Guid id)
        {
            var command = new DeleteProductCommand { Id = id };
            
            try
            {
                var result = await _mediator.Send(command);
                
                if (!result)
                {
                    return NotFound();
                }
                
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting product");
                return BadRequest(ex.Message);
            }
        }
        
        [HttpPost("{id}/publish")]
        [ProducesResponseType((int)HttpStatusCode.NoContent)]
        [ProducesResponseType((int)HttpStatusCode.NotFound)]
        public async Task<IActionResult> PublishProduct(Guid id)
        {
            var command = new PublishProductCommand { Id = id };
            
            try
            {
                var result = await _mediator.Send(command);
                
                if (!result)
                {
                    return NotFound();
                }
                
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error publishing product");
                return BadRequest(ex.Message);
            }
        }
        
        [HttpPost("{id}/reviews")]
        [ProducesResponseType(typeof(ProductReviewDto), (int)HttpStatusCode.Created)]
        [ProducesResponseType((int)HttpStatusCode.BadRequest)]
        [ProducesResponseType((int)HttpStatusCode.NotFound)]
        public async Task<ActionResult<ProductReviewDto>> AddReview(Guid id, [FromBody] AddProductReviewCommand command)
        {
            if (id != command.ProductId)
            {
                return BadRequest("Product ID in URL does not match ID in request body");
            }
            
            try
            {
                var review = await _mediator.Send(command);
                
                if (review == null)
                {
                    return NotFound();
                }
                
                return CreatedAtRoute("GetProduct", new { id = id }, review);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding review");
                return BadRequest(ex.Message);
            }
        }
    }
}
