import 'package:cactus/models/tools.dart';
import '../services/rag_service.dart';
import 'document_similarity_tool.dart';

/// Abstract interface for all tools
abstract class ToolHandler {
  CactusTool get definition;
  
  Future<String> call(Map<String, dynamic> args, RAGService? rag);
}

/// Tool Registry - manages all available tools
class ToolRegistry {
  static final List<ToolHandler> _tools = [
    DocumentSimilarityTool(),
    // Add more tools here as needed
  ];

  static List<CactusTool> getAllDefinitions() {
    return _tools.map((t) => t.definition).toList();
  }

  static Future<String> executeTool(
    String name,
    Map<String, dynamic> args, {
    RAGService? ragService,
  }) async {
    final tool = _tools.firstWhere(
      (t) => t.definition.name == name,
      orElse: () => throw Exception('Tool not found: $name'),
    );
    
    return await tool.call(args, ragService);
  }

  static bool hasTool(String name) {
    return _tools.any((t) => t.definition.name == name);
  }
}