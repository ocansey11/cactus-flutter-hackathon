import 'package:cactus/cactus.dart';

class ModelManager {
  static final Map<String, bool> _downloadedModels = {};
  static final Map<String, bool> _downloadedVoiceModels = {};
  
  static final Map<String, CactusLM> _initializedLLMs = {};
  static final Map<String, CactusSTT> _initializedSTTs = {};
  

  static Future<CactusLM> getOrInitializeLLM({
    required String modelName,
    Function(double?, String, bool)? progressCallback,
  }) async {
    if (_initializedLLMs.containsKey(modelName)) {
      if (progressCallback != null) {
        progressCallback(1.0, 'Using cached model: $modelName', false);
      }
      return _initializedLLMs[modelName]!;
    }
    
    final model = CactusLM();
    
    await ensureLLMDownloaded(
      model: model,
      modelName: modelName,
      progressCallback: progressCallback,
    );
    
    if (progressCallback != null) {
      progressCallback(null, 'Initializing $modelName...', false);
    }
    
    await model.initializeModel(
      params: CactusInitParams(model: modelName),
    );
    
    _initializedLLMs[modelName] = model;
    
    if (progressCallback != null) {
      progressCallback(1.0, 'Model ready: $modelName', false);
    }
    
    return model;
  }

  static Future<void> ensureLLMDownloaded({
    required CactusLM model,
    required String modelName,
    Function(double?, String, bool)? progressCallback,
  }) async {
    if (_downloadedModels[modelName] == true) {
      if (progressCallback != null) {
        progressCallback(1.0, 'Model already downloaded: $modelName', false);
      }
      return;
    }
    
    await model.downloadModel(
      model: modelName,
      downloadProcessCallback: progressCallback,
    );
    
    _downloadedModels[modelName] = true;
  }
  
  static CactusLM? getCachedLLM(String modelName) {
    return _initializedLLMs[modelName];
  }
  
  static Future<void> unloadLLM(String modelName) async {
    if (_initializedLLMs.containsKey(modelName)) {
      _initializedLLMs[modelName]!.unload();
      _initializedLLMs.remove(modelName);
    }
  }
  
  
  static Future<CactusSTT> getOrInitializeSTT({
    required String modelName,
    TranscriptionProvider provider = TranscriptionProvider.whisper,
    Function(double?, String, bool)? progressCallback,
  }) async {
    final cacheKey = '$provider:$modelName';
    
    if (_initializedSTTs.containsKey(cacheKey)) {
      if (progressCallback != null) {
        progressCallback(1.0, 'Using cached STT: $modelName', false);
      }
      return _initializedSTTs[cacheKey]!;
    }
    
    final stt = CactusSTT(provider: provider);
    
    await ensureVoiceModelDownloaded(
      stt: stt,
      modelName: modelName,
      progressCallback: progressCallback,
    );
    
    if (progressCallback != null) {
      progressCallback(null, 'Initializing STT: $modelName...', false);
    }
    
    await stt.init(model: modelName);
    
    _initializedSTTs[cacheKey] = stt;
    
    if (progressCallback != null) {
      progressCallback(1.0, 'STT ready: $modelName', false);
    }
    
    return stt;
  }
  
  static Future<void> ensureVoiceModelDownloaded({
    required CactusSTT stt,
    required String modelName,
    Function(double?, String, bool)? progressCallback,
  }) async {
    if (_downloadedVoiceModels[modelName] == true) {
      if (progressCallback != null) {
        progressCallback(1.0, 'Voice model already downloaded: $modelName', false);
      }
      return;
    }
    
    final isDownloaded = await stt.isModelDownloaded(modelName: modelName);
    
    if (isDownloaded) {
      _downloadedVoiceModels[modelName] = true;
      if (progressCallback != null) {
        progressCallback(1.0, 'Voice model already on disk: $modelName', false);
      }
      return;
    }
    
    await stt.download(
      model: modelName,
      downloadProcessCallback: progressCallback,
    );
    
    _downloadedVoiceModels[modelName] = true;
  }
  
  static CactusSTT? getCachedSTT(String modelName, TranscriptionProvider provider) {
    final cacheKey = '$provider:$modelName';
    return _initializedSTTs[cacheKey];
  }
  
  static void disposeSTT(String modelName, TranscriptionProvider provider) {
    final cacheKey = '$provider:$modelName';
    if (_initializedSTTs.containsKey(cacheKey)) {
      _initializedSTTs[cacheKey]!.dispose();
      _initializedSTTs.remove(cacheKey);
    }
  }
  
  // ============================================================================
  // Future: VLM (Vision Language Models) - Placeholder
  // ============================================================================
  
  // TODO: Add VLM support when Cactus adds vision models
  // static Future<CactusVLM> getOrInitializeVLM({...}) async { }
  
  // ============================================================================
  // Utility Methods
  // ============================================================================
  
  /// Reset all caches (useful for testing or clearing memory)
  static void reset() {
    _downloadedModels.clear();
    _downloadedVoiceModels.clear();
    _initializedLLMs.clear();
    _initializedSTTs.clear();
  }
  
  /// Get cache statistics
  static Map<String, dynamic> getCacheStats() {
    return {
      'downloaded_models': _downloadedModels.length,
      'downloaded_voice_models': _downloadedVoiceModels.length,
      'initialized_llms': _initializedLLMs.length,
      'initialized_stts': _initializedSTTs.length,
      'model_names': _initializedLLMs.keys.toList(),
      'stt_names': _initializedSTTs.keys.toList(),
    };
  }
  
  /// Unload all models and clear cache
  static Future<void> unloadAll() async {
    for (final model in _initializedLLMs.values) {
      model.unload();
    }
    for (final stt in _initializedSTTs.values) {
      stt.dispose();
    }
    reset();
  }
}
