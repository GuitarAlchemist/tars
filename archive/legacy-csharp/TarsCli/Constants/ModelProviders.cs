namespace TarsCli.Constants;

/// <summary>
/// Constants for model providers and models
/// </summary>
public static class ModelProviders
{
    /// <summary>
    /// Anthropic model provider
    /// </summary>
    public static class Anthropic
    {
        /// <summary>
        /// Provider name
        /// </summary>
        public const string Name = "Anthropic";

        /// <summary>
        /// Claude 3 Opus model
        /// </summary>
        public const string Claude3Opus = "claude-3-opus-20240229";

        /// <summary>
        /// Claude 3 Sonnet model
        /// </summary>
        public const string Claude3Sonnet = "claude-3-sonnet-20240229";

        /// <summary>
        /// Claude 3 Haiku model
        /// </summary>
        public const string Claude3Haiku = "claude-3-haiku-20240307";
    }

    /// <summary>
    /// OpenAI model provider
    /// </summary>
    public static class OpenAI
    {
        /// <summary>
        /// Provider name
        /// </summary>
        public const string Name = "OpenAI";

        /// <summary>
        /// GPT-4o model
        /// </summary>
        public const string GPT4o = "gpt-4o";

        /// <summary>
        /// GPT-4 Turbo model
        /// </summary>
        public const string GPT4Turbo = "gpt-4-turbo";

        /// <summary>
        /// GPT-3.5 Turbo model
        /// </summary>
        public const string GPT35Turbo = "gpt-3.5-turbo";
    }

    /// <summary>
    /// Meta model provider
    /// </summary>
    public static class Meta
    {
        /// <summary>
        /// Provider name
        /// </summary>
        public const string Name = "Meta";

        /// <summary>
        /// Llama 3 70B model
        /// </summary>
        public const string Llama3_70B = "llama3:70b";

        /// <summary>
        /// Llama 3 70B Instruct model
        /// </summary>
        public const string Llama3_70B_Instruct = "llama3:70b-instruct";

        /// <summary>
        /// Llama 3 8B model
        /// </summary>
        public const string Llama3_8B = "llama3:8b";

        /// <summary>
        /// Llama 3 8B Instruct model
        /// </summary>
        public const string Llama3_8B_Instruct = "llama3:8b-instruct";

        /// <summary>
        /// Llama 3 405B model
        /// </summary>
        public const string Llama3_405B = "llama3:405b";
    }

    /// <summary>
    /// Google model provider
    /// </summary>
    public static class Google
    {
        /// <summary>
        /// Provider name
        /// </summary>
        public const string Name = "Google";

        /// <summary>
        /// Gemini 1.5 Pro model
        /// </summary>
        public const string Gemini15Pro = "gemini-1.5-pro";

        /// <summary>
        /// Gemini 1.5 Flash model
        /// </summary>
        public const string Gemini15Flash = "gemini-1.5-flash";

        /// <summary>
        /// Gemini 1.0 Pro model
        /// </summary>
        public const string Gemini10Pro = "gemini-1.0-pro";
    }
}
