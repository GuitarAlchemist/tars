using System;
using System.Collections.Generic;

namespace ProductCatalog.Domain.Common
{
    public abstract class Entity
    {
        private readonly List<DomainEvent> _domainEvents = new List<DomainEvent>();
        
        public Guid Id { get; protected set; }
        
        public IReadOnlyCollection<DomainEvent> DomainEvents => _domainEvents.AsReadOnly();
        
        public void AddDomainEvent(DomainEvent domainEvent)
        {
            _domainEvents.Add(domainEvent);
        }
        
        public void RemoveDomainEvent(DomainEvent domainEvent)
        {
            _domainEvents.Remove(domainEvent);
        }
        
        public void ClearDomainEvents()
        {
            _domainEvents.Clear();
        }
        
        public override bool Equals(object obj)
        {
            if (obj is not Entity other)
                return false;
                
            if (ReferenceEquals(this, other))
                return true;
                
            if (GetType() != other.GetType())
                return false;
                
            if (Id == Guid.Empty || other.Id == Guid.Empty)
                return false;
                
            return Id == other.Id;
        }
        
        public override int GetHashCode()
        {
            return Id.GetHashCode() * 41;
        }
        
        public static bool operator ==(Entity left, Entity right)
        {
            if (left is null && right is null)
                return true;
                
            if (left is null || right is null)
                return false;
                
            return left.Equals(right);
        }
        
        public static bool operator !=(Entity left, Entity right)
        {
            return !(left == right);
        }
    }
}
