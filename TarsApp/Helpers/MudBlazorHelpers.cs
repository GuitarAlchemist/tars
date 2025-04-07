using MudBlazor;

namespace TarsApp.Helpers
{
    /// <summary>
    /// Helper methods for MudBlazor components
    /// </summary>
    public static class MudBlazorHelpers
    {
        /// <summary>
        /// Creates a DialogOptions object with the backdrop click disabled
        /// </summary>
        /// <returns>DialogOptions with backdrop click disabled</returns>
        public static DialogOptions DisableBackdropClick()
        {
            return new DialogOptions
            {
                // In MudBlazor 8.x, DisableBackdropClick was renamed to BackdropClick with inverted logic
                BackdropClick = false,
                CloseOnEscapeKey = true
            };
        }
    }
}
