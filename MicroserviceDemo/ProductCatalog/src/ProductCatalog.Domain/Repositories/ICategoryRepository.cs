using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using ProductCatalog.Domain.Entities;

namespace ProductCatalog.Domain.Repositories
{
    public interface ICategoryRepository : IRepository<ProductCategory>
    {
        Task<IReadOnlyList<ProductCategory>> GetRootCategoriesAsync(CancellationToken cancellationToken = default);
        Task<IReadOnlyList<ProductCategory>> GetSubcategoriesAsync(Guid parentId, CancellationToken cancellationToken = default);
        Task<ProductCategory> GetCategoryWithSubcategoriesAsync(Guid id, CancellationToken cancellationToken = default);
        Task<IReadOnlyList<ProductCategory>> GetCategoryPathAsync(Guid categoryId, CancellationToken cancellationToken = default);
    }
}
