import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import 'document_similarity_tool.dart';

// Define multiple tools
final tools = [
  /*
  CactusTool(
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: ToolParametersSchema(
      properties: {
        'location': ToolParameter(
            type: 'string', description: 'City name', required: true),
      },
    ),
  ),
  CactusTool(
    name: "get_stock_price",
    description: "Get current stock price for a company",
    parameters: ToolParametersSchema(
      properties: {
        'symbol': ToolParameter(
            type: 'string', description: 'Stock symbol', required: true),
      },
    ),
  ),
  CactusTool(
    name: "send_email",
    description: "Send an email to someone",
    parameters: ToolParametersSchema(
      properties: {
        'to': ToolParameter(
            type: 'string', description: 'Email address', required: true),
        'subject': ToolParameter(
            type: 'string', description: 'Email subject', required: true),
        'body': ToolParameter(
            type: 'string', description: 'Email body', required: true),
      },
    ),
  ),*/
  CactusTool(
    name: "compute_document_similarity",
    description:
        "Analyze similarity relationships between documents in the knowledge base",
    parameters: ToolParametersSchema(
      properties: {
        'threshold': ToolParameter(
            type: 'number',
            description: 'Minimum similarity score to include (0.0-1.0)',
            required: false),
      },
    ),
  ),
];

/// Registry class to execute tools by name
class ToolRegistry {
  static Future<String> executeTool(
    String toolName,
    Map<String, dynamic> arguments, {
    RAGService? ragService,
  }) async {
    switch (toolName) {
      case 'compute_document_similarity':
        return await DocumentSimilarityTool.execute(arguments, ragService);

      default:
        return 'Unknown tool: $toolName';
    }
  }
}
