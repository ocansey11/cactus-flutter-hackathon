import 'package:cactus/cactus.dart';

class FunctionCallingService {
  final CactusLM _model;  // Qwen - handles both understanding and tool calling
  final List<CactusTool> _availableTools;
  
  FunctionCallingService({
    required CactusLM model,
    required List<CactusTool> tools,
  }) : _model = model,
       _availableTools = tools;
  
  Future<FunctionCallResult> analyzeQuery(String query) async {
    // Document similarity is now triggered via the graph icon button
    // Removed keyword-based detection to avoid chat-based triggering
    
    final lowerQuery = query.toLowerCase();
    
    // Only use keyword-based detection for simple project context queries
    // Let AI model handle note creation and paper analysis for flexibility
    final projectKeywords = [
      'how many papers',
      'how many documents',
      'papers do i have',
      'papers have i',
      'list papers',
      'show papers',
      'project statistics',
      'project stats',
      'project info',
      'project details',
      'project notes',
      'project objectives',
      'in this project',
      'for this project',
      'gathered for',
    ];
    
    if (projectKeywords.any((keyword) => lowerQuery.contains(keyword))) {
      // Determine info_type based on query
      String infoType = 'all';
      if (lowerQuery.contains('how many') || lowerQuery.contains('count')) {
        infoType = 'statistics';
      } else if (lowerQuery.contains('papers') || lowerQuery.contains('documents')) {
        infoType = 'papers';
      } else if (lowerQuery.contains('notes')) {
        infoType = 'notes';
      } else if (lowerQuery.contains('objective') || lowerQuery.contains('goal')) {
        infoType = 'notes';
      }
      
      return FunctionCallResult(
        needsTool: true,
        toolName: 'get_project_context',
        parameters: {'info_type': infoType},
      );
    }
    
    // Use Cactus built-in function calling for complex queries
    print('Using Cactus built-in function calling for query: "$query"');
    
    try {
      final response = await _model.generateCompletion(
        messages: [ChatMessage(content: query, role: 'user')],
        params: CactusCompletionParams(
          tools: _availableTools,
          maxTokens: 512,
          temperature: 0.1,
        ),
      );
      
      print('Raw response object:');
      print('  success: ${response.success}');
      print('  response text: ${response.response}');
      print('  toolCalls length: ${response.toolCalls.length}');
      
      if (!response.success) {
        print('❌ Function calling failed!');
        print('   Error response: ${response.response}');
        print('   Full response object: $response');
        return FunctionCallResult(needsTool: false);
      }
      
      print('Model response: ${response.response}');
      print('Tool calls detected: ${response.toolCalls.length}');
      
      if (response.toolCalls.isEmpty) {
        print('⚠️  No tool calls found in response');
        return FunctionCallResult(needsTool: false);
      }
      
      // Take the first tool call
      final toolCall = response.toolCalls.first;
      print('✅ Selected tool: ${toolCall.name}');
      print('   Parameters: ${toolCall.arguments}');
      print('   Tool call object: $toolCall');
      
      return FunctionCallResult(
        needsTool: true,
        toolName: toolCall.name,
        parameters: toolCall.arguments,
      );
    } catch (e, stackTrace) {
      print('❌ Function calling exception caught!');
      print('   Error: $e');
      print('   Stack trace:');
      print('$stackTrace');
      return FunctionCallResult(needsTool: false);
    }
  }
}

class FunctionCallResult {
  final bool needsTool;
  final String? toolName;
  final Map<String, dynamic> parameters;
  
  FunctionCallResult({
    required this.needsTool,
    this.toolName,
    this.parameters = const {},
  });
  
  @override
  String toString() {
    return 'FunctionCallResult(needsTool: $needsTool, toolName: $toolName, parameters: $parameters)';
  }
}
