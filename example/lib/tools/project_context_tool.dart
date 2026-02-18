import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../services/project_service.dart';
import 'tool_handler.dart';

class ProjectContextTool implements ToolHandler {
  @override
  CactusTool get definition => CactusTool(
        name: 'get_project_context',
        description:
            'Get information about the current project: papers, notes, statistics, or overview.',
        parameters: ToolParametersSchema(
          properties: {
            'info_type': ToolParameter(
              type: 'string',
              description: 'overview | papers | notes | statistics | all',
              required: true,
            ),
          },
        ),
      );

  @override
  Future<String> call(
    Map<String, dynamic> args, {
    RAGService? ragService,
    ProjectService? projectService,
    CactusLM? chatModel,
  }) async {
    if (projectService == null || projectService.currentProject == null) {
      return 'No active project. Select or create a project first.';
    }

    final project = projectService.currentProject!;
    final infoType = args['info_type'] ?? 'all';
    final buffer = StringBuffer();

    if (infoType == 'overview' || infoType == 'all') {
      buffer.writeln('Project: ${project.name}');
      if (project.description != null) {
        buffer.writeln('Description: ${project.description}');
      }
      buffer.writeln();
    }

    if (infoType == 'papers' || infoType == 'all') {
      final papers = projectService.getDocuments(project.id);
      buffer.writeln('Papers (${papers.length}):');
      if (papers.isEmpty) {
        buffer.writeln('No papers uploaded yet.');
      } else {
        for (int i = 0; i < papers.length; i++) {
          buffer.writeln('${i + 1}. ${papers[i].fileName}');
        }
      }
      buffer.writeln();
    }

    if (infoType == 'notes' || infoType == 'all') {
      final notes = projectService.getNotes(project.id);
      buffer.writeln('Notes (${notes.length}):');
      if (notes.isEmpty) {
        buffer.writeln('No notes yet.');
      } else {
        for (final note in notes) {
          buffer.writeln('- [${note.noteType}] ${note.title}');
        }
      }
      buffer.writeln();
    }

    if (infoType == 'statistics' || infoType == 'all') {
      final papers = projectService.getDocuments(project.id);
      final notes = projectService.getNotes(project.id);
      buffer.writeln('Statistics:');
      buffer.writeln('Papers: ${papers.length}');
      buffer.writeln('Notes: ${notes.length}');
    }

    return buffer.toString().trim();
  }
}
