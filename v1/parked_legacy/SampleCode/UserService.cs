using System;
using System.Collections.Generic;
using System.IO;

namespace SampleApp.Services
{
    public class UserService
    {
        private List<User> _users = new List<User>();
        
        public User GetUserById(string userId)
        {
            foreach (var user in _users)
            {
                if (user.Id == userId)
                {
                    return user;
                }
            }
            
            return null;
        }
        
        public List<User> GetUsersByRole(string role)
        {
            var result = new List<User>();
            
            foreach (var user in _users)
            {
                if (user.Role == role)
                {
                    result.Add(user);
                }
            }
            
            return result;
        }
        
        public int CalculateAverageAge()
        {
            int sum = 0;
            for (int i = 0; i < _users.Count; i++)
            {
                sum += _users[i].Age;
            }
            return sum / _users.Count;
        }
        
        public void SaveUsers(string filePath)
        {
            string json = Newtonsoft.Json.JsonConvert.SerializeObject(_users);
            File.WriteAllText(filePath, json);
        }
        
        public void LoadUsers(string filePath)
        {
            if (File.Exists(filePath))
            {
                string json = File.ReadAllText(filePath);
                _users = Newtonsoft.Json.JsonConvert.DeserializeObject<List<User>>(json);
            }
        }
        
        public string FormatUserInfo(User user)
        {
            return string.Format("User: {0}, Email: {1}, Age: {2}", user.Name, user.Email, user.Age);
        }
    }
    
    public class User
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Email { get; set; }
        public int Age { get; set; }
        public string Role { get; set; }
        
        public User(string id, string name, string email, int age, string role)
        {
            Id = id;
            Name = name;
            Email = email;
            Age = age;
            Role = role;
        }
        
        public bool IsAdult()
        {
            return Age >= 18;
        }
    }
}
