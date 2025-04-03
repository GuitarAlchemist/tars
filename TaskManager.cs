using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace TaskManager
{
    public enum TaskPriority
    {
        Low,
        Medium,
        High
    }

    public class Task
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Title { get; set; }
        public string Description { get; set; }
        public DateTime? DueDate { get; set; }
        public TaskPriority Priority { get; set; }
        public string Category { get; set; }
        public bool IsCompleted { get; set; }

        public Task(string title, string description, DateTime? dueDate, TaskPriority priority, string category)
        {
            Title = title;
            Description = description;
            DueDate = dueDate;
            Priority = priority;
            Category = category;
            IsCompleted = false;
        }

        public bool IsOverdue()
        {
            return DueDate.HasValue && DueDate.Value < DateTime.Today && !IsCompleted;
        }

        public void MarkAsComplete()
        {
            IsCompleted = true;
        }
    }

    public class Category
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Name { get; set; }
        public string Description { get; set; }

        public Category(string name, string description)
        {
            Name = name;
            Description = description;
        }
    }

    public class TaskManager
    {
        private List<Task> _tasks = new List<Task>();
        private List<Category> _categories = new List<Category>();
        private FileManager _fileManager = new FileManager();

        public TaskManager()
        {
            LoadData();
        }

        public void AddTask(Task task)
        {
            _tasks.Add(task);
            SaveData();
        }

        public void UpdateTask(string taskId, Task updatedTask)
    {
        if (taskId == null)
        {
            throw new ArgumentNullException(nameof(taskId));
        }

            var existingTask = _tasks.Find(t => t.Id == taskId);
            if (existingTask != null)
            {
                existingTask.Title = updatedTask.Title;
                existingTask.Description = updatedTask.Description;
                existingTask.DueDate = updatedTask.DueDate;
                existingTask.Priority = updatedTask.Priority;
                existingTask.Category = updatedTask.Category;
                existingTask.IsCompleted = updatedTask.IsCompleted;
                SaveData();
            }
        }

        public void DeleteTask(string taskId)
    {
        if (taskId == null)
        {
            throw new ArgumentNullException(nameof(taskId));
        }

            var task = _tasks.Find(t => t.Id == taskId);
            if (task != null)
            {
                _tasks.Remove(task);
                SaveData();
            }
        }

        public void MarkTaskAsComplete(string taskId)
    {
        if (taskId == null)
        {
            throw new ArgumentNullException(nameof(taskId));
        }

            var task = _tasks.Find(t => t.Id == taskId);
            if (task != null)
            {
                task.MarkAsComplete();
                SaveData();
            }
        }

        public List<Task> GetAllTasks()
        {
            return _tasks;
        }

        public List<Task> GetTasksByCategory(string category)
        {
            return _tasks.FindAll(t => t.Category == category);
        }

        public List<Task> GetTasksByPriority(TaskPriority priority)
        {
            return _tasks.FindAll(t => t.Priority == priority);
        }

        public List<Task> GetOverdueTasks()
        {
            return _tasks.FindAll(t => t.IsOverdue());
        }

        public void AddCategory(Category category)
        {
            _categories.Add(category);
            SaveData();
        }

        public void UpdateCategory(string categoryId, Category updatedCategory)
    {
        if (categoryId == null)
        {
            throw new ArgumentNullException(nameof(categoryId));
        }

            var existingCategory = _categories.Find(c => c.Id == categoryId);
            if (existingCategory != null)
            {
                existingCategory.Name = updatedCategory.Name;
                existingCategory.Description = updatedCategory.Description;
                SaveData();
            }
        }

        public void DeleteCategory(string categoryId)
    {
        if (categoryId == null)
        {
            throw new ArgumentNullException(nameof(categoryId));
        }

            var category = _categories.Find(c => c.Id == categoryId);
            if (category != null)
            {
                _categories.Remove(category);
                
                // Update tasks with this category
                foreach (var task in _tasks.FindAll(t => t.Category == category.Name))
                {
                    task.Category = null;
                }
                
                SaveData();
            }
        }

        public List<Category> GetAllCategories()
        {
            return _categories;
        }

        private void SaveData()
        {
            var data = new TaskManagerData
            {
                Tasks = _tasks,
                Categories = _categories
            };
            
            _fileManager.SaveData(data);
        }

        private void LoadData()
        {
            var data = _fileManager.LoadData();
            if (data != null)
            {
                _tasks = data.Tasks;
                _categories = data.Categories;
            }
        }
    }

    public class TaskManagerData
    {
        public List<Task> Tasks { get; set; } = new List<Task>();
        public List<Category> Categories { get; set; } = new List<Category>();
    }

    public class FileManager
    {
        private readonly string _filePath = "taskmanager.json";

        public void SaveData(TaskManagerData data)
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };
            
            string jsonString = JsonSerializer.Serialize(data, options);
            File.WriteAllText(_filePath, jsonString);
        }

        public TaskManagerData LoadData()
        {
            if (!File.Exists(_filePath))
            {
                return new TaskManagerData();
            }
            
            string jsonString = File.ReadAllText(_filePath);
            return JsonSerializer.Deserialize<TaskManagerData>(jsonString);
        }
    }
}

