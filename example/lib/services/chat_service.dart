import 'package:cactus/cactus.dart';

class ChatService {
  final CactusLM _chatModel;
  
  ChatService({required CactusLM chatModel}) : _chatModel = chatModel;
  
  Future<String> simpleChat(String query) async {
    final response = await _chatModel.generateCompletion(
      messages: [
        ChatMessage(
          content: 'You are Cactus, a helpful AI assistant.',
          role: 'system',
        ),
        ChatMessage(content: query, role: 'user'),
      ],
      params: CactusCompletionParams(
        maxTokens: 2000,
        temperature: 0.7,
        stopSequences: [],
      ),
    );
    
    if (!response.success) {
      throw Exception('Failed to generate response');
    }
    
    return stripThinkingTags(response.response);
  }
  
  Future<String> ragChat({
    required String query,
    required String context,
    required String systemPrompt,
  }) async {
    final response = await _chatModel.generateCompletion(
      messages: [
        ChatMessage(content: systemPrompt, role: 'system'),
        ChatMessage(content: context, role: 'user'),
      ],
      params: CactusCompletionParams(
        maxTokens: 2000,
        temperature: 0.3,
        stopSequences: [],
      ),
    );
    
    if (!response.success) {
      throw Exception('Failed to generate response');
    }
    
    return stripThinkingTags(response.response);
  }
  
  String stripThinkingTags(String text) {
    final regex = RegExp(
      r'<think>.*?</think>',
      caseSensitive: false,
      dotAll: true,
    );
    var cleaned = text.replaceAll(regex, '');
    
    cleaned = cleaned.replaceAll('<|im_end|>', '');
    cleaned = cleaned.replaceAll('<end_of_turn>', '');
    cleaned = cleaned.replaceAll('<|endoftext|>', '');
    
    return cleaned.trim();
  }
  
  void unload() {
    _chatModel.unload();
  }
}
