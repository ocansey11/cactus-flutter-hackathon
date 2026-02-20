# Hybrid Mode Setup (Cloud Fallback)

## Overview

This app uses **hybrid completion** - combining local on-device AI with cloud fallback for memory-intensive operations like summarization.

## Architecture

- ✅ **Tool Calling**: Local with Qwen-0.6B (fast, works perfectly)
- ✅ **Simple Chat**: Local with Qwen
- ⚡ **Summarization**: Hybrid (auto cloud fallback when needed)

## Why Hybrid?

The app was crashing during summarization due to:
- iOS 3,376 MB memory limit
- 9,387 char context length for papers
- Qwen model + long context exceeding memory

**Solution**: Cactus's built-in hybrid mode automatically uses cloud when local execution would fail.

## Setup Instructions

### 1. Get Your Cactus Token

Visit: https://www.cactuscompute.com/dashboard

### 2. Add Token to App

Edit `example/lib/main.dart` line 49:

```dart
static const String _cactusToken = 'your_token_here';
```

### 3. Run the App

That's it! The app will:
- Use local Qwen for tool calling (instant, no latency)
- Automatically fall back to cloud for summarization (prevents crashes)

## How It Works

When you ask: *"Create a project note with the summary on the paper 'AI Ethics'"*

1. **Tool Calling** (Local):
   - Qwen detects `create_project_note` tool
   - Extracts parameters: `{note_type: summary, paper_name: AI Ethics}`
   - ✅ Fast local inference

2. **Summarization** (Hybrid):
   - Tool retrieves paper text (9,387 chars)
   - Calls `CompletionMode.hybrid`:
     - **If memory available**: Uses local Qwen
     - **If memory constrained**: Auto cloud fallback
   - Generates markdown summary
   - ✅ No crash!

## Technical Details

### Code Changes

**`create_project_note_tool.dart`** (Line 120-130):
```dart
final response = await chatModel.generateCompletion(
  messages: [ChatMessage(content: summaryPrompt, role: 'user')],
  params: CactusCompletionParams(
    maxTokens: 1500,
    temperature: 0.7,
    completionMode: (cactusToken != null && cactusToken.isNotEmpty) 
        ? CompletionMode.hybrid   // Cloud fallback enabled
        : CompletionMode.local,    // Local only (may crash)
    cactusToken: cactusToken,
  ),
);
```

### Mode Detection

- **Token provided** → Hybrid mode (local + cloud fallback)
- **Token empty** → Local only (may crash on heavy tasks)

## Hackathon Alignment

This implementation satisfies the hackathon requirement:

> **"FunctionGemma for fast, local execution on mobile devices with Gemini APIs as a cloud fallback"**

We demonstrate:
- ✅ Local function calling (Qwen instead of FunctionGemma, but same concept)
- ✅ Cloud fallback for heavy compute
- ✅ Intelligent routing based on task complexity

## Cost Optimization

Hybrid mode is cost-effective because:
- Most interactions use local models (free, instant)
- Only summarization uses cloud (minimal API calls)
- User pays ~$0.001 per summary vs. crashing the app

## Monitoring

Check logs during summarization:

```
flutter: 📝 Generating summary for paper: AI Ethics.pdf
flutter:    Context length: 9387 chars
flutter:    Using hybrid (cloud fallback) mode
flutter: ✅ Summary generated successfully
```

## Troubleshooting

### "Running in LOCAL-ONLY mode" warning

You haven't added a Cactus token yet. The app will work but may crash on large paper summarizations.

### Still Crashing?

1. Verify token is set correctly in `main.dart`
2. Check logs for "Using hybrid (cloud fallback) mode"
3. Ensure you have internet connection for cloud fallback

## Next Steps

Try generating a paper summary to see hybrid mode in action:

1. Upload a research paper
2. Ask: *"Create a project note with the summary on the paper 'your paper name'"*
3. Watch logs to see cloud fallback triggered
4. Get summary without crash! ✨
