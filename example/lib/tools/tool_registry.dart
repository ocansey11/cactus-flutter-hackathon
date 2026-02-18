import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../services/project_service.dart';
import 'tool_handler.dart';
import 'document_similarity_tool.dart';
import 'research_paper_tool.dart';
import 'project_context_tool.dart';
import 'create_project_note_tool.dart';

class ToolRegistry {
  final RAGService? ragService;
  final ProjectService? projectService;
  final CactusLM? chatModel;

  final List<ToolHandler> _tools = [
    DocumentSimilarityTool(),
    ResearchPaperTool(),
    ProjectContextTool(),
    CreateProjectNoteTool(),
  ];

  ToolRegistry({
    this.ragService,
    this.projectService,
    this.chatModel,
  });

  List<CactusTool> get definitions => _tools.map((t) => t.definition).toList();

  Future<String?> execute(String name, Map<String, dynamic> args) async {
    final handler = _tools.cast<ToolHandler?>().firstWhere(
      (t) => t!.definition.name == name,
      orElse: () => null,
    );

    if (handler == null) return null;

    return await handler.call(
      args,
      ragService: ragService,
      projectService: projectService,
      chatModel: chatModel,
    );
  }

  bool hasTool(String name) => _tools.any((t) => t.definition.name == name);
}
