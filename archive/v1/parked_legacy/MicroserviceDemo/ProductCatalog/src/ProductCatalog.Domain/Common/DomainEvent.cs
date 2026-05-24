using System;
using MediatR;

namespace ProductCatalog.Domain.Common
{
    public abstract class DomainEvent : INotification
    {
        public DateTime Timestamp { get; }
        
        protected DomainEvent()
        {
            Timestamp = DateTime.UtcNow;
        }
    }
}
