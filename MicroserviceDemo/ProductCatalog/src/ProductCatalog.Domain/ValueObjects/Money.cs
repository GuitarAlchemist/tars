using System;
using System.Collections.Generic;
using ProductCatalog.Domain.Common;

namespace ProductCatalog.Domain.ValueObjects
{
    public class Money : ValueObject
    {
        public decimal Amount { get; private set; }
        public string Currency { get; private set; }
        
        private Money() { }
        
        public Money(decimal amount, string currency)
        {
            if (string.IsNullOrWhiteSpace(currency))
                throw new ArgumentException("Currency cannot be empty", nameof(currency));
                
            if (currency.Length != 3)
                throw new ArgumentException("Currency code must be 3 characters", nameof(currency));
                
            Amount = amount;
            Currency = currency.ToUpperInvariant();
        }
        
        public static Money FromDecimal(decimal amount, string currency)
        {
            return new Money(amount, currency);
        }
        
        public static Money operator +(Money left, Money right)
        {
            if (left.Currency != right.Currency)
                throw new InvalidOperationException("Cannot add money with different currencies");
                
            return new Money(left.Amount + right.Amount, left.Currency);
        }
        
        public static Money operator -(Money left, Money right)
        {
            if (left.Currency != right.Currency)
                throw new InvalidOperationException("Cannot subtract money with different currencies");
                
            return new Money(left.Amount - right.Amount, left.Currency);
        }
        
        public static Money operator *(Money left, decimal right)
        {
            return new Money(left.Amount * right, left.Currency);
        }
        
        public static Money operator /(Money left, decimal right)
        {
            if (right == 0)
                throw new DivideByZeroException();
                
            return new Money(left.Amount / right, left.Currency);
        }
        
        public override string ToString()
        {
            return $"{Amount} {Currency}";
        }
        
        protected override IEnumerable<object> GetEqualityComponents()
        {
            yield return Amount;
            yield return Currency;
        }
    }
}
