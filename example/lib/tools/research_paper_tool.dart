import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../services/project_service.dart';
import 'tool_handler.dart';

class ResearchPaperTool implements ToolHandler {
  @override
  CactusTool get definition => CactusTool(
        name: 'analyze_research_paper',
        description:
            'Analyze and summarize a research paper. Optionally extract a specific section.',
        parameters: ToolParametersSchema(
          properties: {
            'paper_name': ToolParameter(
              type: 'string',
              description: 'Name or partial name of the paper',
              required: true,
            ),
            'section': ToolParameter(
              type: 'string',
              description:
                  'Specific section to extract: background, methodology, results, conclusion',
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
    if (ragService == null || chatModel == null) {
      return 'Required services not available.';
    }

    final paperName = args['paper_name'] as String?;
    final section = args['section'] as String?;

    if (paperName == null || paperName.isEmpty) {
      return 'Paper name is required.';
    }

    final allDocs = await ragService.getAllDocuments();
    final paper = allDocs.firstWhere(
      (d) => d.fileName.toLowerCase().contains(paperName.toLowerCase()),
      orElse: () => throw Exception('Paper "$paperName" not found.'),
    );

    final content = paper.chunks.map((c) => c.content).join('\n\n');

    final prompt = section != null && section.isNotEmpty
        ? _sectionPrompt(paper.fileName, content, _normalizeSection(section))
        : _fullAnalysisPrompt(paper.fileName, content);

    final response = await chatModel.generateCompletion(
      messages: [ChatMessage(content: prompt, role: 'user')],
      params: CactusCompletionParams(maxTokens: 1000, temperature: 0.3),
    );

    if (!response.success) return 'Failed to analyze paper.';
    return response.response;
  }

  String _normalizeSection(String section) {
    final s = section.toLowerCase();
    if (s.contains('method') || s.contains('approach')) return 'methodology';
    if (s.contains('intro') || s.contains('background')) return 'background';
    if (s.contains('result') || s.contains('finding')) return 'results';
    if (s.contains('conclusion') || s.contains('discussion')) return 'conclusion';
    return section;
  }

  String _sectionPrompt(String name, String content, String section) =>
      'Extract and summarize the $section section from "$name".\n\nContent:\n$content\n\nBe concise and use bullet points.';

  String _fullAnalysisPrompt(String name, String content) =>
      'Analyze "$name" with sections: Background, Methodology, Results, Conclusions.\n\nContent:\n$content\n\nUse markdown headers.';
}
