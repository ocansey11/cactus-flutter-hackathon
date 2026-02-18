// tool_handler.dart
// Kept as a separate file intentionally to avoid circular imports.
// tool_registry.dart imports the tool files, and the tool files need ToolHandler —
// if ToolHandler lived in tool_registry.dart that would be a circular dependency.
// Keeping it here breaks the cycle: tools import tool_handler, registry imports both.

import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../services/project_service.dart';

abstract class ToolHandler {
  CactusTool get definition;

  Future<String> call(
    Map<String, dynamic> args, {
    RAGService? ragService,
    ProjectService? projectService,
    CactusLM? chatModel,
  });
}
