import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';

class ResearchPaperTool {
  static Future<String> execute(
    Map<String, dynamic> arguments,
    RAGService? ragService,
    CactusLM? chatModel,
  ) async {
    if (ragService == null || chatModel == null) {
      return 'RAG service or chat model not available';
    }

    final paperName = arguments['paper_name'] as String?;
    final section = arguments['section'] as String?; // optional: background, methodology, results, etc.

    if (paperName == null || paperName.isEmpty) {
      return 'Please specify a paper name to analyze';
    }

    try {
      // Get the specific document
      final allDocs = await ragService.getAllDocuments();
      final paper = allDocs.firstWhere(
        (doc) => doc.fileName.toLowerCase().contains(paperName.toLowerCase()),
        orElse: () => throw Exception('Paper "$paperName" not found'),
      );

      // Get all chunks for this paper
      final chunks = paper.chunks.toList();
      final fullContent = chunks.map((c) => c.content).join('\n\n');

      // If specific section requested, extract that
      if (section != null && section.isNotEmpty) {
        // Normalize section names
        final normalizedSection = _normalizeSection(section);
        
        return await _extractSection(
          fullContent: fullContent,
          section: normalizedSection,
          chatModel: chatModel,
          paperName: paper.fileName,
        );
      }

      // Otherwise, generate full structured analysis
      return await _generateFullAnalysis(
        fullContent: fullContent,
        chatModel: chatModel,
        paperName: paper.fileName,
      );
    } catch (e) {
      return 'Error analyzing paper: $e';
    }
  }

  /// Normalize section names to standard terms
  static String _normalizeSection(String section) {
    final normalized = section.toLowerCase().trim();
    
    // Map common variations to standard section names
    if (normalized.contains('approach') || normalized.contains('method')) {
      return 'methodology';
    }
    if (normalized.contains('intro') || normalized.contains('background')) {
      return 'background';
    }
    if (normalized.contains('result') || normalized.contains('finding')) {
      return 'results';
    }
    if (normalized.contains('conclusion') || normalized.contains('discussion')) {
      return 'conclusion';
    }
    if (normalized.contains('related') || normalized.contains('prior')) {
      return 'related work';
    }
    
    return section;
  }

  static Future<String> _extractSection({
    required String fullContent,
    required String section,
    required CactusLM chatModel,
    required String paperName,
  }) async {
    final sectionNote = section.toLowerCase() == 'methodology' 
        ? '(also called approach, methods, or experimental setup)'
        : '';
    
    final prompt = '''Analyze this research paper and extract information about the ${section.toUpperCase()} section $sectionNote.

Paper: $paperName

Content:
$fullContent

Instructions:
1. Find and summarize the ${section.toLowerCase()} section
2. Be concise but comprehensive  
3. Use bullet points for key information
4. If this section is not present, say so
5. Focus on the main techniques, algorithms, or procedures used

Provide the ${section.toLowerCase()} summary:''';

    final response = await chatModel.generateCompletion(
      messages: [ChatMessage(content: prompt, role: 'user')],
      params: CactusCompletionParams(
        maxTokens: 500,
        temperature: 0.3,
      ),
    );

    if (!response.success) {
      return 'Failed to extract section from the paper';
    }

    return '**${section.toUpperCase()} - $paperName**\n\n${response.response}';
  }

  static Future<String> _generateFullAnalysis({
    required String fullContent,
    required CactusLM chatModel,
    required String paperName,
  }) async {
    final prompt = '''Analyze this research paper and provide a structured summary with the following sections:

Paper: $paperName

Content:
$fullContent

Instructions:
Create a comprehensive summary with these sections:

1. **Title & Authors**: Extract if present
2. **Background/Introduction**: What problem does this paper address? Why is it important?
3. **Related Work**: What previous research is mentioned?
4. **Methodology**: How did they approach the problem? What methods/models/techniques?
5. **Key Results**: What were the main findings?
6. **Conclusions**: What are the takeaways?

Format each section clearly with headers. Be concise but thorough.''';

    final response = await chatModel.generateCompletion(
      messages: [ChatMessage(content: prompt, role: 'user')],
      params: CactusCompletionParams(
        maxTokens: 1000,
        temperature: 0.3,
      ),
    );

    if (!response.success) {
      return 'Failed to generate analysis for the paper';
    }

    return '**Research Paper Analysis**\n\n${response.response}';
  }

  /// Ask a specific question about a paper
  static Future<String> askQuestion({
    required String paperName,
    required String question,
    required RAGService ragService,
    required CactusLM chatModel,
  }) async {
    try {
      // Get the specific document
      final allDocs = await ragService.getAllDocuments();
      final paper = allDocs.firstWhere(
        (doc) => doc.fileName.toLowerCase().contains(paperName.toLowerCase()),
        orElse: () => throw Exception('Paper "$paperName" not found'),
      );

      // Search for relevant chunks
      final results = await ragService.search(query: question, limit: 5);
      
      // Filter to only this paper's chunks
      final paperChunks = results
          .where((r) => r.chunk.document.target?.id == paper.id)
          .toList();
      
      if (paperChunks.isEmpty) {
        return 'Could not find relevant information in "$paperName" to answer: $question';
      }

      final context = ragService.buildContext(paperChunks);
      
      final prompt = '''Based on the research paper "$paperName", answer this question:

Context from paper:
$context

Question: $question

Provide a clear, evidence-based answer using only the information from the paper above.''';

      final response = await chatModel.generateCompletion(
        messages: [ChatMessage(content: prompt, role: 'user')],
        params: CactusCompletionParams(
          maxTokens: 400,
          temperature: 0.3,
        ),
      );

      if (!response.success) {
        return 'Failed to answer the question about this paper';
      }

      return response.response;
    } catch (e) {
      return 'Error: $e';
    }
  }
}
