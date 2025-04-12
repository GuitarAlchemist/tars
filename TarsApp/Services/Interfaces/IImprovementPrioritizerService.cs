using System.Collections.Generic;
using System.Threading.Tasks;
using TarsApp.ViewModels;

namespace TarsApp.Services.Interfaces
{
    public interface IImprovementPrioritizerService
    {
        Task<List<string>> GetImprovementCategoriesAsync();
        Task<List<string>> GetImprovementPrioritiesAsync();
        Task<List<string>> GetImprovementTagsAsync();
        Task<bool> PrioritizeImprovementAsync(string executionId, string improvementId, int priority);
        Task<bool> TagImprovementAsync(string executionId, string improvementId, string tag);
        Task<bool> RemoveTagFromImprovementAsync(string executionId, string improvementId, string tag);
    }
}
