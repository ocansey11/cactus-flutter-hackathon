import 'dart:convert';
import 'package:cactus/cactus.dart';

class FunctionCallingService {
  final CactusLM _functionModel;
  
  FunctionCallingService({required CactusLM functionModel})
    : _functionModel = functionModel;
  
  Future<FunctionCallResult> analyzeQuery(String query) async {
    // Document similarity is now triggered via the graph icon button
    // Removed keyword-based detection to avoid chat-based triggering
    
    // Try AI-based detection for other tools
    final prompt = _buildFunctionDetectionPrompt(query);
    
    try {
      final response = await _functionModel.generateCompletion(
        messages: [
          ChatMessage(content: prompt, role: 'user'),
        ],
        params: CactusCompletionParams(
          maxTokens: 150,
          temperature: 0.3,
          stopSequences: [],
        ),
      );
      
      if (!response.success) {
        return FunctionCallResult(needsTool: false);
      }
      
      final result = _parseFunctionCall(response.response);
      print('🔧 AI function detection result: needsTool=${result.needsTool}, tool=${result.toolName}');
      return result;
    } catch (e) {
      print('🔧 Function detection error: $e');
      return FunctionCallResult(needsTool: false);
    }
  }
  
  String _buildFunctionDetectionPrompt(String query) {
    return '''Analyze if this query needs a tool/function call.

Available tools:
(Note: Document similarity is accessed via the graph icon, not chat)

Query: "$query"

Instructions:
- If no tool needed, respond with: {"needs_tool": false}
- Be precise with parameter extraction

Response format (JSON only, no explanation):
{
  "needs_tool": true/false,
  "tool_name": "tool_name" or null,
  "parameters": {"param": "value"} or {}
}

Response:''';
  }
  
  FunctionCallResult _parseFunctionCall(String response) {
    try {
      String cleaned = response.trim();
      
      if (cleaned.startsWith('```json')) {
        cleaned = cleaned.substring(7);
      }
      if (cleaned.startsWith('```')) {
        cleaned = cleaned.substring(3);
      }
      if (cleaned.endsWith('```')) {
        cleaned = cleaned.substring(0, cleaned.length - 3);
      }
      cleaned = cleaned.trim();
      
      final jsonStart = cleaned.indexOf('{');
      final jsonEnd = cleaned.lastIndexOf('}');
      
      if (jsonStart != -1 && jsonEnd != -1) {
        final jsonStr = cleaned.substring(jsonStart, jsonEnd + 1);
        final json = jsonDecode(jsonStr);
        
        final needsTool = json['needs_tool'] == true;
        final toolName = json['tool_name'] as String?;
        final parameters = json['parameters'] as Map<String, dynamic>? ?? {};
        
        return FunctionCallResult(
          needsTool: needsTool,
          toolName: toolName,
          parameters: parameters,
        );
      }
      
      return FunctionCallResult(needsTool: false);
    } catch (e) {
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
