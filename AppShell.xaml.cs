using CameraMaui.Pages;

namespace CameraMaui
{
    public partial class AppShell : Shell
    {
        public AppShell()
        {
            InitializeComponent();

            // Register routes for navigation
            Routing.RegisterRoute(nameof(TemplateCreatorPage), typeof(TemplateCreatorPage));
        }
    }
}
