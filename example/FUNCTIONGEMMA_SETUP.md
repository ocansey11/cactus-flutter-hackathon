# FunctionGemma Setup for Hackathon

## Overview
This hackathon requires using **FunctionGemma** for local tool calling. Cactus doesn't currently have FunctionGemma in its official model catalog, so we need to add it manually.

## Finding FunctionGemma GGUF Files

### Option 1: Official Google Sources
Check these Hugging Face repositories for GGUF files:
- https://huggingface.co/google/gemma-2b-it (look for function calling variants)
- https://huggingface.co/bartowski (often has quantized GGUF conversions)
- Search Hugging Face for "functiongemma gguf" or "gemma-2b function calling gguf"

### Option 2: Use LM Studio or Similar Tools
1. Download FunctionGemma in GGUF format using LM Studio
2. Copy the .gguf file to: `~/Library/Application Support/cactus-flutter-hackathon/models/function-gemma-2b/`

## Manual Download Steps

### 1. Find the GGUF URL
Look for quantized versions (recommended for mobile):
- **Q4_K_M** (~1.2GB) - Good balance of quality and size
- **Q8_0** (~2GB) - Higher quality, larger size

### 2. Update the Downloader
Edit `example/lib/services/function_gemma_downloader.dart` and replace the placeholder URL with the actual GGUF download link:

```dart
static const String functionGemmaUrl = 'YOUR_ACTUAL_GGUF_URL_HERE';
static const String modelFileName = 'your-actual-filename.gguf';
```

### 3. Download via UI
Run the app and use the FunctionGemma download page (if created), or manually place the file in:
```
${AppDocuments}/models/function-gemma-2b/model.gguf
```

### 4. Update main.dart to Use FunctionGemma

Replace Qwen with FunctionGemma:

```dart
// Initialize FunctionGemma instead of Qwen
await _chatModel.initializeModel(
  params: CactusInitParams(
    model: 'function-gemma-2b',  // Our custom model
    contextSize: 2048,
  )
);

// Update FunctionCallingService
_functionService = FunctionCallingService(
  model: _chatModel,
  tools: tools,
);
```

## Verifying the Setup

1. Check if model is loaded:
```dart
print('Model loaded: ${_chatModel.isLoaded()}');
```

2. Test function calling:
```dart
final result = await _functionService.analyzeQuery(
  'write a summary for the paper: AI Ethics and Fairness'
);
print('Tool detected: ${result.toolName}');
```

## Expected Model Specs

**FunctionGemma 2B:**
- Size: 1-2GB (quantized)
- Format: GGUF
- Trained specifically for function calling
- Better JSON output than standard Gemma

## Troubleshooting

### Model Won't Load
- Verify the .gguf file is in the correct directory
- Check file isn't corrupted (compare file size with expected)
- Ensure filename matches what's in the code

### Tool Calling Fails
- FunctionGemma should be much better at JSON than standard Gemma
- If still seeing malformed JSON, check the prompt format
- Verify you're using the built-in Cactus function calling system

## Alternative: Contact Hackathon Organizers

Since this is a sponsored hackathon, you may want to ask:
1. **Cactus team**: Can you add FunctionGemma to the official model catalog?
2. **DeepMind team**: What's the recommended way to get FunctionGemma GGUF?
3. **Hackathon organizers**: Do you have a special build or download link?

## Hybrid Architecture (Backup Plan)

If FunctionGemma proves difficult to obtain, the hybrid approach still aligns with hackathon goals:
- **Local**: Qwen3-0.6B for function calling (works reasonably well)
- **Cloud**: Gemini API for complex reasoning
- **Demo**: Show intelligent routing between edge and cloud

This demonstrates the core hackathon concept even without FunctionGemma specifically.
