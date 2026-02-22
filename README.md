# Cactus Flutter Hackathon — Local AI Assistant

A privacy-first, offline AI assistant built on the [Cactus](https://github.com/cactus-compute/cactus) Flutter framework. Built before the Cactus x Deep Mind function gemma hackathon in UCL London

To work with local models you need these basic features persistent conversation memory, RAG document search, voice transcription, and tool calling — all running entirely on-device with no cloud dependencies. We have setup this as a preparation to the hackathon and hopefully build an actual end to end service around these foudndations.

---

## Getting Started

### 1. Clone and switch to the latest branch

git clone https://github.com/ocansey11/cactus-flutter-hackathon.git
cd cactus-flutter-hackathon
git checkout merging-core

> `merging-core` is the most up to datte working branch. It contains persistent memory, tool calling, and the message router.

### 2. Install dependencies

flutter pub get

### 3. Run the app

- `cd example`
- `flutter pub get`
- `flutter run`

The app will prompt you to download the model on first launch (~500MB). After that, everything runs offline.

---

## Running on Android

You need a physical Android device or emulator with API level 24+.

### Enable USB Debugging
1. Go to Settings → About Phone
2. Tap Build Number 7 times to unlock Developer Options
3. Go to Settings → Developer Options → enable USB Debugging
4. Connect via USB and accept the prompt on your phone

### Permissions — add to android/app/src/main/AndroidManifest.xml

<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />

### Run

- `flutter devices`       # confirm your device shows up
- `flutter run`          # deploys to connected Android device

---

## Project Structure

lib/                        # Core package
  services/                 # Cactus wrappers, RAG, memory
  models/                   # Data models and ObjectBox entities

example/lib/                # App UI
  pages/                    # Screens (chat, voice, documents)
  services/                 # Message router, conversation service
  tools/                    # Tool definitions and handlers
  prompts/                  # Prompt engineering templates
  widgets/                  # Shared UI components

---

## Branches

| Branch | Description |
|---|---|
| `merging-core` | Latest — persistent memory, tool calling, message router |
| `memory-persistance` | ObjectBox persistent conversations |
| `speech-feature` | Voice transcription with Whisper |
| `tools-structure` | Tool calling architecture |
| `hackathoFeb` | Original hackathon baseline |

---

## Support

- [Cactus Docs](https://cactuscompute.com/docs)
- [Discord](https://discord.gg/bNurx3AXTJ)
- [Models on Hugging Face](https://huggingface.co/Cactus-Compute/models)
