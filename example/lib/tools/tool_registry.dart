import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import 'document_similarity_tool.dart';
import 'research_paper_tool.dart';
import 'project_context_tool.dart';
import 'create_project_note_tool.dart';

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
  CactusTool(
    name: "analyze_research_paper",
    description:
        "Analyze and summarize a research paper with structured sections (background, methodology, results, etc.)",
    parameters: ToolParametersSchema(
      properties: {
        'paper_name': ToolParameter(
            type: 'string',
            description: 'Name or partial name of the paper to analyze',
            required: true),
        'section': ToolParameter(
            type: 'string',
            description:
                'Specific section to extract (optional): background, methodology, results, conclusion',
            required: false),
      },
    ),
  ),
  CactusTool(
    name: "get_project_context",
    description:
        "Get information about the current research project including paper count, notes, objectives, and project details. Use when user asks about the current project, papers gathered, project goals, or statistics.",
    parameters: ToolParametersSchema(
      properties: {
        'info_type': ToolParameter(
            type: 'string',
            description: 'Type of information to retrieve: overview, papers, notes, statistics, or all',
            required: true),
        'note_type': ToolParameter(
            type: 'string',
            description: 'Filter notes by type (concept, summary, writeup, general)',
            required: false),
      },
    ),
  ),
  CactusTool(
    name: "create_project_note",
    description:
        "Create and save a note for the current project. Can generate paper summaries, project objectives, or project plans. Use when user asks to create/write a summary, set objectives, or make a plan.",
    parameters: ToolParametersSchema(
      properties: {
        'note_type': ToolParameter(
            type: 'string',
            description: 'Type of note: summary, objective, plan, or general',
            required: true),
        'paper_name': ToolParameter(
            type: 'string',
            description: 'Paper name for summaries',
            required: false),
        'content': ToolParameter(
            type: 'string',
            description: 'User content for objectives/plans/general notes',
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
    CactusLM? chatModel,
    ProjectService? projectService,
  }) async {
    switch (toolName) {
      case 'compute_document_similarity':
        return await DocumentSimilarityTool.execute(arguments, ragService);

      case 'analyze_research_paper':
        return await ResearchPaperTool.execute(arguments, ragService, chatModel);

      case 'get_project_context':
        return await ProjectContextTool.execute(arguments, projectService, ragService);

      case 'create_project_note':
        return await CreateProjectNoteTool.execute(arguments, projectService, ragService, chatModel);

      default:
        return 'Unknown tool: $toolName';
    }
  }
}
