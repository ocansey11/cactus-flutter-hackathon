import 'package:cactus/cactus.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'model_manager.dart';

class VoiceService {
  late CactusSTT _stt;
  final FlutterTts _tts = FlutterTts();
  
  Future<SpeechRecognitionResult?>? _currentTranscription;
  bool _isModelReady = false;
  
  Future<void> initialize({String model = 'whisper-base'}) async {
    // Get or initialize STT model (downloads + initializes + caches)
    _stt = await ModelManager.getOrInitializeSTT(
      modelName: model,
      provider: TranscriptionProvider.whisper,
      progressCallback: (progress, status, isError) {
        if (isError) {
          print("STT error: $status");
        } else if (progress != null) {
          print("STT downloading: ${(progress * 100).toStringAsFixed(1)}%");
        } else {
          print("STT status: $status");
        }
      },
    );
    
    // Setup TTS
    await _tts.setLanguage("en-US");
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
    
    _isModelReady = true;
  }
  
  Future<void> startRecording() async {
    if (!_isModelReady) {
      throw Exception('VoiceService not initialized');
    }
    
    if (_currentTranscription != null) {
      throw Exception('Already recording');
    }
    
    final params = SpeechRecognitionParams(
      sampleRate: 16000,
      maxDuration: 30000,
    );
    
    _currentTranscription = _stt.transcribe(params: params);
  }
  
  Future<String> stopRecording() async {
    if (_currentTranscription == null) {
      return '';
    }
    
    _stt.stop();
    
    final result = await _currentTranscription!;
    _currentTranscription = null;
    
    if (result != null && result.success) {
      return result.text;
    }
    
    return '';
  }
  
  bool get isRecording => _stt.isRecording;
  
  bool get isReady => _isModelReady;
  
  Future<void> speak(String text) async {
    if (!_isModelReady) {
      throw Exception('VoiceService not initialized');
    }
    
    await _tts.speak(text);
  }
  
  Future<void> stopSpeaking() async {
    await _tts.stop();
  }
  
  void dispose() {
    // STT is managed by ModelManager, don't dispose here
    // Just stop any ongoing operations
    if (_currentTranscription != null) {
      _stt.stop();
    }
  }
}
