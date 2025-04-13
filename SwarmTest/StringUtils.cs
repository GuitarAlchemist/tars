// This is a test file for auto-coding
// The code has been improved by TARS

using System;
using System.Text;

namespace SwarmTest
{
    public static class StringUtils
    {
        /// <summary>
        /// Reverses a string.
        /// </summary>
        /// <param name="input">The string to reverse</param>
        /// <returns>The reversed string</returns>
        public static string Reverse(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }
            
            char[] charArray = input.ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }
        
        /// <summary>
        /// Checks if a string is a palindrome (reads the same forward and backward).
        /// </summary>
        /// <param name="input">The string to check</param>
        /// <returns>True if the string is a palindrome, false otherwise</returns>
        public static bool IsPalindrome(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return true;
            }
            
            // Remove spaces and convert to lowercase for a more lenient check
            string normalized = input.Replace(" ", "").ToLower();
            
            int left = 0;
            int right = normalized.Length - 1;
            
            while (left < right)
            {
                if (normalized[left] != normalized[right])
                {
                    return false;
                }
                
                left++;
                right--;
            }
            
            return true;
        }
        
        /// <summary>
        /// Counts the number of words in a string.
        /// </summary>
        /// <param name="input">The string to count words in</param>
        /// <returns>The number of words in the string</returns>
        public static int CountWords(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return 0;
            }
            
            // Split by whitespace and count non-empty parts
            return input.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
        }
        
        /// <summary>
        /// Truncates a string to a specified length and adds an ellipsis if truncated.
        /// </summary>
        /// <param name="input">The string to truncate</param>
        /// <param name="maxLength">The maximum length of the string</param>
        /// <param name="ellipsis">The ellipsis to add if truncated (default: "...")</param>
        /// <returns>The truncated string</returns>
        public static string Truncate(string input, int maxLength, string ellipsis = "...")
        {
            if (string.IsNullOrEmpty(input) || input.Length <= maxLength)
            {
                return input;
            }
            
            return input.Substring(0, maxLength - ellipsis.Length) + ellipsis;
        }
    }
}
