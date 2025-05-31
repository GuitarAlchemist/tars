using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.Common;

namespace TarsEngine.Services.Core.Base
{
    /// <summary>
    /// Base implementation for all services in the TarsEngine.
    /// </summary>
    public abstract class ServiceBase : IService
    {
        /// <summary>
        /// The logger instance.
        /// </summary>
        protected readonly ILogger Logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="ServiceBase"/> class.
        /// </summary>
        /// <param name="logger">The logger instance.</param>
        protected ServiceBase(ILogger logger)
        {
            Logger = logger;
        }

        /// <inheritdoc/>
        public abstract string Name { get; }
    }
}
