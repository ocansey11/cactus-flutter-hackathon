import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';

class ProjectContextTool {
  static const String name = 'get_project_context';
  static const String description = 
      'Get information about the current research project including paper count, '
      'notes, objectives, and other project details. Use this when the user asks '
      'about the current project, papers gathered, project goals, or project statistics.';

  static Map<String, dynamic> get parameters => {
    'type': 'object',
    'properties': {
      'info_type': {
        'type': 'string',
        'description': 'Type of information to retrieve',
        'enum': ['overview', 'papers', 'notes', 'statistics', 'all'],
      },
      'note_type': {
        'type': 'string',
        'description': 'Filter notes by type (only used when info_type is "notes")',
        'enum': ['concept', 'summary', 'writeup', 'general'],
      },
    },
    'required': ['info_type'],
  };

  static Future<String> execute(
    Map<String, dynamic> params,
    ProjectService? projectService,
    RAGService? ragService,
  ) async {
    if (projectService == null || projectService.currentProject == null) {
      return 'No project is currently active. Please select or create a project first.';
    }

    final currentProject = projectService.currentProject!;
    final infoType = params['info_type'] ?? 'all';
    final noteTypeFilter = params['note_type'];

    final buffer = StringBuffer();

    try {
      if (infoType == 'overview' || infoType == 'all') {
        buffer.writeln('PROJECT INFORMATION:');
        buffer.writeln('Name: ${currentProject.name}');
        if (currentProject.description?.isNotEmpty == true) {
          buffer.writeln('Description: ${currentProject.description}');
        }
        buffer.writeln('Created: ${_formatDate(currentProject.createdAt)}');
        buffer.writeln('Last Updated: ${_formatDate(currentProject.updatedAt)}');
        buffer.writeln();
      }

      if (infoType == 'statistics' || infoType == 'all') {
        final stats = projectService.getProjectStats(currentProject.id);
        buffer.writeln('PROJECT STATISTICS:');
        buffer.writeln('Total Papers: ${stats.documentCount}');
        buffer.writeln('Concept Notes: ${stats.conceptNotes}');
        buffer.writeln('Summary Notes: ${stats.summaryNotes}');
        buffer.writeln('Writeup Notes: ${stats.writeupNotes}');
        buffer.writeln('Total Notes: ${stats.noteCount}');
        buffer.writeln();
      }

      if (infoType == 'papers' || infoType == 'all') {
        final papers = await projectService.storageService.listPapers(
          currentProject.name,
        );
        buffer.writeln('PAPERS (${papers.length}):');
        if (papers.isEmpty) {
          buffer.writeln('No papers have been uploaded to this project yet.');
        } else {
          for (int i = 0; i < papers.length; i++) {
            buffer.writeln('${i + 1}. ${papers[i]}');
          }
        }
        buffer.writeln();
      }

      if (infoType == 'notes' || infoType == 'all') {
        final allNotes = projectService.getProjectNotes(currentProject.id);
        
        List<ProjectNote> notesToShow;
        if (noteTypeFilter != null) {
          notesToShow = allNotes.where((n) => n.noteType == noteTypeFilter).toList();
          buffer.writeln('${noteTypeFilter.toUpperCase()} NOTES (${notesToShow.length}):');
        } else {
          notesToShow = allNotes;
          buffer.writeln('ALL NOTES (${notesToShow.length}):');
        }

        if (notesToShow.isEmpty) {
          buffer.writeln('No notes found.');
        } else {
          for (final note in notesToShow) {
            buffer.writeln('Title: ${note.title}');
            buffer.writeln('Type: ${note.noteType}');
            buffer.writeln('Content: ${note.content}');
            buffer.writeln('Created: ${_formatDate(note.createdAt)}');
            buffer.writeln('---');
          }
        }
      }

      return buffer.toString().trim();
    } catch (e) {
      return 'Error retrieving project context: $e';
    }
  }

  static String _formatDate(DateTime date) {
    return '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
  }
}
