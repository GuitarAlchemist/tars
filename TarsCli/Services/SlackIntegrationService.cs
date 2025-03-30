using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for integrating with Slack to post announcements and updates
    /// </summary>
    public class SlackIntegrationService
    {
        private readonly ILogger<SlackIntegrationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly SecretsService _secretsService;
        private readonly HttpClient _httpClient;
        private readonly string _defaultChannel;
        private string? _webhookUrl;
        private bool _isEnabled;

        public SlackIntegrationService(
            ILogger<SlackIntegrationService> logger,
            IConfiguration configuration,
            SecretsService secretsService)
        {
            _logger = logger;
            _configuration = configuration;
            _secretsService = secretsService;
            _httpClient = new HttpClient();

            // Get default channel from configuration
            _defaultChannel = _configuration["Slack:DefaultChannel"] ?? "#tars";

            // Initialize the service
            InitializeAsync().GetAwaiter().GetResult();
        }

        /// <summary>
        /// Initialize the Slack integration service
        /// </summary>
        private async Task InitializeAsync()
        {
            try
            {
                // Get webhook URL from secrets
                _webhookUrl = await _secretsService.GetSecretAsync("Slack:WebhookUrl");

                // If not found in secrets, try configuration
                if (string.IsNullOrEmpty(_webhookUrl))
                {
                    _webhookUrl = _configuration["Slack:WebhookUrl"];
                }

                // Check if Slack integration is enabled
                _isEnabled = !string.IsNullOrEmpty(_webhookUrl);

                if (_isEnabled)
                {
                    _logger.LogInformation("Slack integration is enabled");
                }
                else
                {
                    _logger.LogInformation("Slack integration is disabled (no webhook URL configured)");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing Slack integration");
                _isEnabled = false;
            }
        }

        /// <summary>
        /// Check if Slack integration is enabled
        /// </summary>
        /// <returns>True if enabled, false otherwise</returns>
        public bool IsEnabled()
        {
            return _isEnabled;
        }

        /// <summary>
        /// Set the webhook URL for Slack integration
        /// </summary>
        /// <param name="webhookUrl">Webhook URL</param>
        /// <returns>True if successful, false otherwise</returns>
        public async Task<bool> SetWebhookUrlAsync(string webhookUrl)
        {
            try
            {
                // Validate the webhook URL
                if (!Uri.TryCreate(webhookUrl, UriKind.Absolute, out _) ||
                    !webhookUrl.StartsWith("https://hooks.slack.com/"))
                {
                    _logger.LogError("Invalid Slack webhook URL");
                    return false;
                }

                // Save the webhook URL to secrets
                await _secretsService.SetSecretAsync("Slack:WebhookUrl", webhookUrl);

                // Update the webhook URL
                _webhookUrl = webhookUrl;
                _isEnabled = true;

                _logger.LogInformation("Slack webhook URL set successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error setting Slack webhook URL");
                return false;
            }
        }

        /// <summary>
        /// Post a message to Slack
        /// </summary>
        /// <param name="message">Message text</param>
        /// <param name="channel">Channel to post to (optional)</param>
        /// <returns>True if successful, false otherwise</returns>
        public async Task<bool> PostMessageAsync(string message, string? channel = null)
        {
            if (!_isEnabled || string.IsNullOrEmpty(_webhookUrl))
            {
                _logger.LogWarning("Slack integration is not enabled");
                return false;
            }

            try
            {
                // Create the message payload
                var payload = new
                {
                    text = message,
                    channel = channel ?? _defaultChannel
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_webhookUrl, content);

                // Check the response
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation($"Message posted to Slack channel {channel ?? _defaultChannel}");
                    return true;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error posting to Slack: {response.StatusCode} - {errorContent}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error posting message to Slack");
                return false;
            }
        }

        /// <summary>
        /// Post an announcement to Slack
        /// </summary>
        /// <param name="title">Announcement title</param>
        /// <param name="message">Announcement message</param>
        /// <param name="channel">Channel to post to (optional)</param>
        /// <returns>True if successful, false otherwise</returns>
        public async Task<bool> PostAnnouncementAsync(string title, string message, string? channel = null)
        {
            if (!_isEnabled || string.IsNullOrEmpty(_webhookUrl))
            {
                _logger.LogWarning("Slack integration is not enabled");
                return false;
            }

            try
            {
                // Create the message payload with blocks for better formatting
                var payload = new
                {
                    channel = channel ?? _defaultChannel,
                    blocks = new object[]
                    {
                        new
                        {
                            type = "header",
                            text = new
                            {
                                type = "plain_text",
                                text = $"📢 {title}",
                                emoji = true
                            }
                        },
                        new
                        {
                            type = "divider"
                        },
                        new
                        {
                            type = "section",
                            text = new
                            {
                                type = "mrkdwn",
                                text = message
                            }
                        },
                        new
                        {
                            type = "context",
                            elements = new object[]
                            {
                                new
                                {
                                    type = "mrkdwn",
                                    text = $"*Posted by:* TARS | *Date:* {DateTime.Now:yyyy-MM-dd HH:mm:ss}"
                                }
                            }
                        }
                    }
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_webhookUrl, content);

                // Check the response
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation($"Announcement posted to Slack channel {channel ?? _defaultChannel}");
                    return true;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error posting announcement to Slack: {response.StatusCode} - {errorContent}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error posting announcement to Slack");
                return false;
            }
        }

        /// <summary>
        /// Post a feature update to Slack
        /// </summary>
        /// <param name="featureName">Feature name</param>
        /// <param name="description">Feature description</param>
        /// <param name="channel">Channel to post to (optional)</param>
        /// <returns>True if successful, false otherwise</returns>
        public async Task<bool> PostFeatureUpdateAsync(string featureName, string description, string? channel = null)
        {
            if (!_isEnabled || string.IsNullOrEmpty(_webhookUrl))
            {
                _logger.LogWarning("Slack integration is not enabled");
                return false;
            }

            try
            {
                // Create the message payload with blocks for better formatting
                var payload = new
                {
                    channel = channel ?? _defaultChannel,
                    blocks = new object[]
                    {
                        new
                        {
                            type = "header",
                            text = new
                            {
                                type = "plain_text",
                                text = $"🚀 New Feature: {featureName}",
                                emoji = true
                            }
                        },
                        new
                        {
                            type = "divider"
                        },
                        new
                        {
                            type = "section",
                            text = new
                            {
                                type = "mrkdwn",
                                text = description
                            }
                        },
                        new
                        {
                            type = "context",
                            elements = new object[]
                            {
                                new
                                {
                                    type = "mrkdwn",
                                    text = $"*Posted by:* TARS | *Date:* {DateTime.Now:yyyy-MM-dd HH:mm:ss}"
                                }
                            }
                        }
                    }
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_webhookUrl, content);

                // Check the response
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation($"Feature update posted to Slack channel {channel ?? _defaultChannel}");
                    return true;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error posting feature update to Slack: {response.StatusCode} - {errorContent}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error posting feature update to Slack");
                return false;
            }
        }

        /// <summary>
        /// Post a milestone to Slack
        /// </summary>
        /// <param name="milestoneName">Milestone name</param>
        /// <param name="description">Milestone description</param>
        /// <param name="channel">Channel to post to (optional)</param>
        /// <returns>True if successful, false otherwise</returns>
        public async Task<bool> PostMilestoneAsync(string milestoneName, string description, string? channel = null)
        {
            if (!_isEnabled || string.IsNullOrEmpty(_webhookUrl))
            {
                _logger.LogWarning("Slack integration is not enabled");
                return false;
            }

            try
            {
                // Create the message payload with blocks for better formatting
                var payload = new
                {
                    channel = channel ?? _defaultChannel,
                    blocks = new object[]
                    {
                        new
                        {
                            type = "header",
                            text = new
                            {
                                type = "plain_text",
                                text = $"🏆 Milestone Achieved: {milestoneName}",
                                emoji = true
                            }
                        },
                        new
                        {
                            type = "divider"
                        },
                        new
                        {
                            type = "section",
                            text = new
                            {
                                type = "mrkdwn",
                                text = description
                            }
                        },
                        new
                        {
                            type = "context",
                            elements = new object[]
                            {
                                new
                                {
                                    type = "mrkdwn",
                                    text = $"*Posted by:* TARS | *Date:* {DateTime.Now:yyyy-MM-dd HH:mm:ss}"
                                }
                            }
                        }
                    }
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_webhookUrl, content);

                // Check the response
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation($"Milestone posted to Slack channel {channel ?? _defaultChannel}");
                    return true;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error posting milestone to Slack: {response.StatusCode} - {errorContent}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error posting milestone to Slack");
                return false;
            }
        }

        /// <summary>
        /// Post an auto-improvement update to Slack
        /// </summary>
        /// <param name="improvementCount">Number of improvements</param>
        /// <param name="details">Improvement details</param>
        /// <param name="channel">Channel to post to (optional)</param>
        /// <returns>True if successful, false otherwise</returns>
        public async Task<bool> PostAutoImprovementUpdateAsync(int improvementCount, string details, string? channel = null)
        {
            if (!_isEnabled || string.IsNullOrEmpty(_webhookUrl))
            {
                _logger.LogWarning("Slack integration is not enabled");
                return false;
            }

            try
            {
                // Create the message payload with blocks for better formatting
                var payload = new
                {
                    channel = channel ?? _defaultChannel,
                    blocks = new object[]
                    {
                        new
                        {
                            type = "header",
                            text = new
                            {
                                type = "plain_text",
                                text = $"🤖 Auto-Improvement Update: {improvementCount} Improvements",
                                emoji = true
                            }
                        },
                        new
                        {
                            type = "divider"
                        },
                        new
                        {
                            type = "section",
                            text = new
                            {
                                type = "mrkdwn",
                                text = details
                            }
                        },
                        new
                        {
                            type = "context",
                            elements = new object[]
                            {
                                new
                                {
                                    type = "mrkdwn",
                                    text = $"*Posted by:* TARS | *Date:* {DateTime.Now:yyyy-MM-dd HH:mm:ss}"
                                }
                            }
                        }
                    }
                };

                // Serialize the payload
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    Encoding.UTF8,
                    "application/json");

                // Send the request
                var response = await _httpClient.PostAsync(_webhookUrl, content);

                // Check the response
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation($"Auto-improvement update posted to Slack channel {channel ?? _defaultChannel}");
                    return true;
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError($"Error posting auto-improvement update to Slack: {response.StatusCode} - {errorContent}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error posting auto-improvement update to Slack");
                return false;
            }
        }
    }
}
