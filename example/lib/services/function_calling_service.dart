import 'dart:convert';
import 'package:cactus/cactus.dart';
import '../prompts/prompts.dart';

class FunctionCallingService {
  final CactusLM _queryAnalyzer;  // Qwen - smart query understanding
  final CactusLM _functionModel;  // Gemma - lightweight function dispatcher
  
  FunctionCallingService({
    required CactusLM queryAnalyzer,
    required CactusLM functionModel,
  }) : _queryAnalyzer = queryAnalyzer,
       _functionModel = functionModel;
  
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
    
    // Two-stage AI function detection:
    // Stage 1: Qwen analyzes complex query and extracts intent
    // Stage 2: Gemma matches simplified intent to tool
    print('Running two-stage AI function detection for query: "$query"');
    
    try {
      // STAGE 1: Qwen simplifies the query
      final simplifiedIntent = await _simplifyQuery(query);
      print('Qwen simplified intent: $simplifiedIntent');
      
      if (simplifiedIntent == null) {
        return FunctionCallResult(needsTool: false);
      }
      
      // STAGE 2: Gemma matches to tool
      final toolMatch = await _matchToTool(simplifiedIntent);
      print('Gemma tool match: needsTool=${toolMatch.needsTool}, tool=${toolMatch.toolName}, params=${toolMatch.parameters}');
      return toolMatch;
    } catch (e) {
      print('Function detection error: $e');
      return FunctionCallResult(needsTool: false);
    }
  }
  
  Future<String?> _simplifyQuery(String query) async {
    final prompt = ToolDispatchPrompts.querySimplify(query: query);

    try {
      final response = await _queryAnalyzer.generateCompletion(
        messages: [ChatMessage(content: prompt, role: 'user')],
        params: CactusCompletionParams(
          maxTokens: 1000,
          temperature: 0.1,
          // No stop sequences - let it generate fully including thinking
        ),
      );
      
      if (!response.success) {
        return null;
      }
      
      // Clean up response
      String cleaned = response.response.trim();
      print('Qwen raw response: "$cleaned"');
      
      // Remove thinking process - extract only the output after </think>
      if (cleaned.contains('<think>')) {
        final thinkEndIndex = cleaned.indexOf('</think>');
        if (thinkEndIndex != -1) {
          // Take everything after </think>
          cleaned = cleaned.substring(thinkEndIndex + '</think>'.length).trim();
        } else {
          // Has <think> but no closing tag - try to extract after <think>
          final thinkStartIndex = cleaned.indexOf('<think>');
          if (thinkStartIndex != -1) {
            cleaned = cleaned.substring(thinkStartIndex + '<think>'.length).trim();
          }
        }
      }
      
      // Take only the first line
      cleaned = cleaned.split('\n').first.trim();
      
      // Remove any remaining markdown, tags, or prefixes
      cleaned = cleaned.replaceAll('```', '')
                       .replaceAll('Simplified:', '')
                       .replaceAll('Output:', '')
                       .replaceAll('<|endoftext|>', '')
                       .replaceAll('<|im_end|>', '')
                       .replaceAll('<|im_start|>', '')
                       .trim();
      
      print('Qwen simplified intent: $cleaned');
      return cleaned.isEmpty ? null : cleaned;
    } catch (e) {
      return null;
    }
  }
  
  Future<FunctionCallResult> _matchToTool(String simplifiedIntent) async {
    final prompt = ToolDispatchPrompts.toolMatch(intent: simplifiedIntent);

    try {
      final response = await _functionModel.generateCompletion(
        messages: [ChatMessage(content: prompt, role: 'user')],
        params: CactusCompletionParams(
          maxTokens: 150,
          temperature: 0.1,
        ),
      );
      
      if (!response.success) {
        return FunctionCallResult(needsTool: false);
      }
      
      print('Gemma raw response: "${response.response}"');
      return _parseFunctionCall(response.response);
    } catch (e) {
      print('Gemma matching error: $e');
      return FunctionCallResult(needsTool: false);
    }
  }
  
  FunctionCallResult _parseFunctionCall(String response) {
    try {
      // Clean up the response
      String jsonStr = response.trim();
      
      // Remove markdown code blocks
      jsonStr = jsonStr.replaceAll('```json', '').replaceAll('```', '');
      
      // Remove JavaScript-style comments
      jsonStr = jsonStr.replaceAll(RegExp(r'//.*?(?=\n|$)'), '');
      
      // Remove special tokens
      jsonStr = jsonStr.replaceAll('<end_of_turn>', '')
                       .replaceAll('<|im_end|>', '')
                       .replaceAll('<|im_start|>', '');
      
      // Extract JSON object (handles nested braces)
      final openBrace = jsonStr.indexOf('{');
      if (openBrace != -1) {
        int braceCount = 0;
        int closeBrace = -1;
        for (int i = openBrace; i < jsonStr.length; i++) {
          if (jsonStr[i] == '{') braceCount++;
          if (jsonStr[i] == '}') braceCount--;
          if (braceCount == 0) {
            closeBrace = i;
            break;
          }
        }
        if (closeBrace != -1) {
          jsonStr = jsonStr.substring(openBrace, closeBrace + 1);
        }
      }
      
      // Fix common malformed patterns
      // Fix extra quote-brace at end: }"}  -> }}
      jsonStr = jsonStr.replaceAll(RegExp(r'\}"\}$'), '}}');
      
      // Remove trailing commas before closing braces
      jsonStr = jsonStr.replaceAll(RegExp(r',\s*}'), '}');
      
      jsonStr = jsonStr.trim();
      print('Cleaned JSON: $jsonStr');
      
      final parsed = jsonDecode(jsonStr) as Map<String, dynamic>;
      final functionName = parsed['function'] as String?;
      
      if (functionName == null || functionName == 'none') {
        return FunctionCallResult(needsTool: false);
      }
      
      final parameters = parsed['parameters'] as Map<String, dynamic>? ?? {};
      final confidence = (parsed['confidence'] as num?)?.toDouble() ?? 0.5;

      // Below threshold — treat as no-tool to avoid wrong dispatch
      if (confidence < 0.7) {
        print('Gemma confidence $confidence below threshold — skipping tool');
        return FunctionCallResult(needsTool: false);
      }

      return FunctionCallResult(
        needsTool: true,
        toolName: functionName,
        parameters: parameters,
        confidence: confidence,
      );
    } catch (e) {
      print('JSON parsing error: $e');
      print('Failed to parse: $response');
      return FunctionCallResult(needsTool: false);
    }
  }
}

class FunctionCallResult {
  final bool needsTool;
  final String? toolName;
  final Map<String, dynamic> parameters;
  final double confidence;

  FunctionCallResult({
    required this.needsTool,
    this.toolName,
    this.parameters = const {},
    this.confidence = 1.0,
  });

  @override
  String toString() =>
      'FunctionCallResult(needsTool: $needsTool, toolName: $toolName, '
      'confidence: $confidence, parameters: $parameters)';
}
