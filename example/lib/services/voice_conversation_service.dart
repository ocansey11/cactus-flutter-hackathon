import 'voice_service.dart';
import 'conversation_service.dart';

class VoiceConversationService {
  final VoiceService voiceService;
  final ConversationService conversationService;
  
  VoiceConversationService({
    required this.voiceService,
    required this.conversationService,
  });
  
  Future<String> processVoiceInput(String transcription) async {
    if (transcription.isEmpty) {
      return "I didn't hear anything. Please try again.";
    }
    
    final response = await conversationService.handleQuery(
      query: transcription,
    );
    
    return response;
  }
  
  Future<void> handleFullVoiceInteraction(String transcription) async {
    final response = await processVoiceInput(transcription);
    await voiceService.speak(response);
  }
}
