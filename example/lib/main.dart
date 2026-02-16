import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'pages/rag_chat.dart';
import 'pages/voice_chat.dart';
import 'services/conversation_service.dart';
import 'services/chat_service.dart';
import 'services/rag_service.dart';
import 'services/function_calling_service.dart';
import 'services/message_router.dart';
import 'services/model_manager.dart';
import 'package:cactus/memory/document_metadata_store.dart';
import 'package:cactus/memory/similarity_cache.dart';
import 'package:cactus/memory/conversation_store.dart';
import 'package:cactus/memory/objectbox_manager.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Hackathon',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.light,
        ),
      ),
      home: const MainPage(),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  final _rag = CactusRAG();
  
  late CactusLM _embeddingModel;
  late CactusLM _functionModel;
  late CactusLM _chatModel;
  late ConversationService _conversationService;
  late ConversationStore _conversationStore;
  late String _currentConversationId;
  
  bool _isInitialized = false;
  bool _isVoiceMode = false;
  String _statusMessage = 'Initializing...';

  @override
  void initState() {
    super.initState();
    CactusTelemetry.setTelemetryToken('a83c7f7a-43ad-4823-b012-cbeb587ae788');
    _initializeServices();
  }

  @override
  void dispose() {
    _rag.close();
    _conversationStore.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    try {
      const embeddingModelName = 'qwen3-0.6';  
      const chatModelName = 'qwen3-0.6';  
      const functionModelName = 'gemma3-270m';
      
      setState(() => _statusMessage = 'Loading Qwen embedding model (600MB)...');
      _embeddingModel = await ModelManager.getOrInitializeLLM(
        modelName: embeddingModelName,
        progressCallback: (progress, status, isError) {
          setState(() {
            if (isError) {
              _statusMessage = 'Error: $status';
            } else {
              _statusMessage = status;
            }
          });
        },
      );

      setState(() => _statusMessage = 'Loading Qwen chat model (600MB)...');
      _chatModel = await ModelManager.getOrInitializeLLM(
        modelName: chatModelName,
        progressCallback: (progress, status, isError) {
          setState(() {
            if (isError) {
              _statusMessage = 'Error: $status';
            } else {
              _statusMessage = status;
            }
          });
        },
      );

      setState(() => _statusMessage = 'Loading Gemma function model (270MB)...');
      _functionModel = await ModelManager.getOrInitializeLLM(
        modelName: functionModelName,
        progressCallback: (progress, status, isError) {
          setState(() {
            if (isError) {
              _statusMessage = 'Error: $status';
            } else {
              _statusMessage = status;
            }
          });
        },
      );

      setState(() => _statusMessage = 'Setting up services...');
      
      final metadataStore = DocumentMetadataStore();
      final similarityCache = SimilarityCache();
      
      final ragService = RAGService(
        rag: _rag,
        embeddingModel: _embeddingModel,
        metadataStore: metadataStore,
        similarityCache: similarityCache,
      );
      await ragService.initialize();
      
      await ObjectBoxManager.initialize(_rag);
      
      _conversationStore = ConversationStore(
        embeddingModel: _embeddingModel,
      );
      await _conversationStore.loadFromDisk();
      
      final existingConversations = _conversationStore.getAllConversations();
      if (existingConversations.isEmpty) {
        final defaultConvo = await _conversationStore.createConversation(
          title: 'Main Chat',
          isVoiceMode: false,
        );
        _currentConversationId = defaultConvo.id;
      } else {
        _currentConversationId = existingConversations.first.id;
      }

      final chatService = ChatService(
        chatModel: _chatModel,
      );

      final functionService = FunctionCallingService(
        functionModel: _functionModel,
      );

      final messageRouter = MessageRouter(
        rag: _rag,
        functionService: functionService,
      );

      _conversationService = ConversationService(
        ragService: ragService,
        chatService: chatService,
        messageRouter: messageRouter,
      );

      setState(() {
        _isInitialized = true;
        _statusMessage = 'Ready! (Qwen: embeddings + chat, Gemma: tool calling)';
      });
    } catch (e) {
      setState(() => _statusMessage = 'Initialization failed: $e');
    }
  }

  void _toggleMode() {
    setState(() => _isVoiceMode = !_isVoiceMode);
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Text(
                  _statusMessage,
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return _isVoiceMode
        ? VoiceChatPage(
            conversationService: _conversationService,
            conversationStore: _conversationStore,
            currentConversationId: _currentConversationId,
            onSwitchMode: _toggleMode,
          )
        : RAGChatPage(
            conversationService: _conversationService,
            conversationStore: _conversationStore,
            currentConversationId: _currentConversationId,
            onSwitchMode: _toggleMode,
          );
  }
}