#include <Momo/Core/Application.h>
#include <MomoEditor/EditorLayer.h>
#include <cstring>

// The standalone scene editor.
//   --scene=<path>  opens a scene file (.json) or a single glTF file
int main(int argc, char** argv)
{
    Momo::ApplicationSpecification spec;
    spec.Name = "Momo Editor";
    spec.WindowSpec.Title = "Momo Editor";
    spec.WindowSpec.Width = 1920;
    spec.WindowSpec.Height = 1080;
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], "--scene=", 8) == 0 && argv[i][8] != '\0')
            spec.ScenePath = argv[i] + 8;

    Momo::Application app(spec);
    app.PushLayer<Momo::Editor::EditorLayer>();
    app.Run();
    app.Shutdown();
    return 0;
}
