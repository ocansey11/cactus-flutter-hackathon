/// Configuration file for API keys and secrets
/// 
/// IMPORTANT: This file (config.example.dart) is a TEMPLATE.
/// Copy it to config.dart and add your actual API keys there.
/// The config.dart file is gitignored and will never be committed.

class AppConfig {
  /// Cactus Cloud Token for hybrid mode fallback
  /// 
  /// Get your token from: https://www.cactuscompute.com/dashboard
  /// 
  /// Cactus provides hybrid execution - local on-device AI with automatic
  /// cloud fallback when memory constraints are hit or tasks are too heavy.
  /// 
  /// How it works:
  /// - Tool calling: Always local (fast, no latency)
  /// - Summarization: Hybrid (cloud fallback when memory exceeded)
  /// 
  /// When `CompletionMode.hybrid` is enabled and local execution would
  /// fail (e.g., 9,387 char context + Qwen exceeds 3,376 MB iOS limit),
  /// Cactus automatically falls back to cloud inference.
  /// 
  /// Leave empty to use local-only mode (may crash on memory-intensive tasks).
  static const String cactusToken = 'YOUR_CACTUS_TOKEN_HERE';
}
