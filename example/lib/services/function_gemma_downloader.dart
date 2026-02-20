import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/foundation.dart';

/// Helper service to download and manage FunctionGemma model for hackathon
class FunctionGemmaDownloader {
  // FunctionGemma 2B quantized GGUF from Hugging Face
  // This is a placeholder - we need to find the correct URL
  static const String functionGemmaUrl = 'https://huggingface.co/google/gemma-2b-it-function-calling/resolve/main/gemma-2b-it-function-calling-q8_0.gguf';
  
  static const String modelSlug = 'function-gemma-2b';
  static const String modelFileName = 'gemma-2b-it-function-calling-q8_0.gguf';
  
  /// Check if FunctionGemma is already downloaded
  static Future<bool> isDownloaded() async {
    final appDocDir = await getApplicationDocumentsDirectory();
    final modelPath = '${appDocDir.path}/models/$modelSlug';
    final modelDir = Directory(modelPath);
    
    if (await modelDir.exists()) {
      final files = await modelDir.list().toList();
      return files.any((file) => file.path.endsWith('.gguf'));
    }
    return false;
  }
  
  /// Get the path to the FunctionGemma model
  static Future<String> getModelPath() async {
    final appDocDir = await getApplicationDocumentsDirectory();
    return '${appDocDir.path}/models/$modelSlug';
  }
  
  /// Download FunctionGemma from Hugging Face
  static Future<bool> download({
    Function(double, String)? onProgress,
  }) async {
    try {
      if (await isDownloaded()) {
        debugPrint('FunctionGemma already downloaded');
        onProgress?.call(1.0, 'Already downloaded');
        return true;
      }
      
      final appDocDir = await getApplicationDocumentsDirectory();
      final modelsDir = Directory('${appDocDir.path}/models');
      await modelsDir.create(recursive: true);
      
      final modelDir = Directory('${appDocDir.path}/models/$modelSlug');
      await modelDir.create(recursive: true);
      
      final filePath = '${modelDir.path}/$modelFileName';
      final client = HttpClient();
      
      debugPrint('Downloading FunctionGemma from $functionGemmaUrl');
      onProgress?.call(0.0, 'Starting download...');
      
      final request = await client.getUrl(Uri.parse(functionGemmaUrl));
      final response = await request.close();
      
      if (response.statusCode != 200) {
        throw Exception('Failed to download: HTTP ${response.statusCode}');
      }
      
      final file = File(filePath);
      final sink = file.openWrite();
      
      final contentLength = response.contentLength;
      int totalBytes = 0;
      
      await for (final chunk in response) {
        sink.add(chunk);
        totalBytes += chunk.length;
        
        if (contentLength > 0) {
          final progress = totalBytes / contentLength;
          final mbDownloaded = totalBytes / (1024 * 1024);
          final mbTotal = contentLength / (1024 * 1024);
          onProgress?.call(
            progress,
            'Downloaded ${mbDownloaded.toStringAsFixed(1)} / ${mbTotal.toStringAsFixed(1)} MB',
          );
        }
      }
      
      await sink.close();
      client.close();
      
      debugPrint('FunctionGemma downloaded successfully to $filePath');
      onProgress?.call(1.0, 'Download complete');
      return true;
      
    } catch (e) {
      debugPrint('Error downloading FunctionGemma: $e');
      onProgress?.call(0.0, 'Download failed: $e');
      return false;
    }
  }
  
  /// Get available FunctionGemma download URLs
  /// Note: These are placeholder URLs - need to find actual GGUF files
  static List<Map<String, String>> getDownloadOptions() {
    return [
      {
        'name': 'FunctionGemma 2B (Q8_0 - ~2GB)',
        'url': 'https://huggingface.co/google/gemma-2b-it-function-calling/resolve/main/gemma-2b-it-function-calling-q8_0.gguf',
        'filename': 'gemma-2b-it-function-calling-q8_0.gguf',
        'size': '2GB',
        'quantization': 'Q8_0',
      },
      {
        'name': 'FunctionGemma 2B (Q4_K_M - ~1.2GB)',
        'url': 'https://huggingface.co/google/gemma-2b-it-function-calling/resolve/main/gemma-2b-it-function-calling-q4_k_m.gguf',
        'filename': 'gemma-2b-it-function-calling-q4_k_m.gguf',
        'size': '1.2GB',
        'quantization': 'Q4_K_M',
      },
    ];
  }
}
