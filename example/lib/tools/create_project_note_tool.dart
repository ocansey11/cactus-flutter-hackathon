import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../services/project_service.dart';
import 'tool_handler.dart';

class CreateProjectNoteTool implements ToolHandler {
  @override
  CactusTool get definition => CactusTool(
        name: 'create_project_note',
        description:
            'Create and save a note for the current project. Use for summaries, objectives, plans, or general notes.',
        parameters: ToolParametersSchema(
          properties: {
            'note_type': ToolParameter(
              type: 'string',
              description: 'summary | objective | plan | general',
              required: true,
            ),
            'paper_name': ToolParameter(
              type: 'string',
              description: 'Paper to summarize (required for summary type)',
              required: false,
            ),
            'content': ToolParameter(
              type: 'string',
              description: 'Content for objective, plan, or general notes',
              required: false,
            ),
            'conversation_id': ToolParameter(
              type: 'string',
              description: 'ID of the current conversation',
              required: false,
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
      return 'No active project.';
    }
    if (chatModel == null || ragService == null) {
      return 'Required services not available.';
    }

    final project = projectService.currentProject!;
    final noteType = args['note_type'] as String;
    final paperName = args['paper_name'] as String?;
    final userContent = args['content'] as String?;
    final conversationId = args['conversation_id'] as String? ?? 'unknown';

    String title;
    String content;
    List<String> referencedPapers = [];

    if (noteType == 'summary') {
      if (paperName == null || paperName.isEmpty) {
        return 'Paper name required for summary.';
      }

      final results = await ragService.search(
        query: paperName,
        projectName: project.name,
        limit: 10,
      );

      if (results.isEmpty) return 'Paper "$paperName" not found in project.';

      referencedPapers = [paperName];
      final context = results.map((r) => r.chunk.content).join('\n\n');

      final response = await chatModel.generateCompletion(
        messages: [
          ChatMessage(
            content:
                'Summarize "$paperName" in markdown with sections: Overview, Key Objectives, Methodology, Findings, Conclusions.\n\nContent:\n$context',
            role: 'user',
          )
        ],
        params: CactusCompletionParams(maxTokens: 1500, temperature: 0.7),
      );

      if (!response.success) return 'Failed to generate summary.';
      title = 'Summary: $paperName';
      content = response.response;
    } else {
      if (userContent == null || userContent.isEmpty) {
        return 'Content required for $noteType note.';
      }

      final prompts = {
        'objective':
            'Format this project objective in markdown with: Goals, Background, Expected Outcomes.\n\n$userContent',
        'plan':
            'Format this project plan in markdown with: Overview, Phases, Milestones, Success Criteria.\n\n$userContent',
        'general': null,
      };

      if (noteType == 'general') {
        title = 'Note';
        content = userContent;
      } else {
        final response = await chatModel.generateCompletion(
          messages: [ChatMessage(content: prompts[noteType]!, role: 'user')],
          params: CactusCompletionParams(maxTokens: 1200, temperature: 0.7),
        );
        if (!response.success) return 'Failed to generate $noteType note.';
        title = noteType == 'objective' ? 'Project Objective' : 'Project Plan';
        content = response.response;
      }
    }

    projectService.createNote(
      projectId: project.id,
      conversationId: conversationId,
      title: title,
      content: content,
      noteType: noteType,
      referencedPapers: referencedPapers,
    );

    return 'Note "$title" saved to ${project.name}.';
  }
}
