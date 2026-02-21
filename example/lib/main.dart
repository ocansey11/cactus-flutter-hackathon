import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'config.dart';
import 'pages/projects_page.dart';
import 'services/conversation_service.dart';
import 'services/chat_service.dart';
import 'services/rag_service.dart';
import 'services/function_calling_service.dart';
import 'services/message_router.dart';
import 'services/model_manager.dart';
import 'tools/tool_registry.dart';

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
  late CactusLM _chatModel;
  late CactusLM _functionModel;  // Qwen for tool calling (reuses chat model)
  late ConversationService _conversationService;
  ProjectService? _projectService;

  bool _isInitialized = false;
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
    super.dispose();
  }

  Future<void> _initializeServices() async {
    try {
      setState(
          () => _statusMessage = 'Loading Qwen model (600MB)...');
      _embeddingModel = await ModelManager.getOrInitializeLLM(
        modelName: 'qwen3-0.6',
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

      setState(() => _statusMessage = 'Configuring Qwen for chat...');
      _chatModel = await ModelManager.getOrInitializeLLM(
        modelName: 'qwen3-0.6',
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

      setState(() => _statusMessage = 'Using Qwen for tool calling...');
      _functionModel = _chatModel;  // Reuse Qwen model for both chat and tools
      setState(() => _statusMessage = 'Qwen configured for tool calling...');

      setState(() => _statusMessage = 'Setting up services...');

      final ragService = RAGService(
        rag: _rag,
        embeddingModel: _embeddingModel,
      );
      await ragService.initialize();

      final functionService = FunctionCallingService(
        model: _functionModel,  // Qwen handles tool calling
        tools: tools,           // Tools from tool_registry.dart
      );

      final chatService = ChatService(
        chatModel: _chatModel,
      );

      final messageRouter = MessageRouter(
        rag: _rag,
        functionService: functionService,
      );

      _projectService = ProjectService(
        store: _rag.store,
      );
      await _projectService!.initialize();

      _conversationService = ConversationService(
        chatService: chatService,
        ragService: ragService,
        messageRouter: messageRouter,
        projectService: _projectService,
        cactusToken: AppConfig.cactusToken.isEmpty ? null : AppConfig.cactusToken,
      );

      // Log hybrid mode status
      if (AppConfig.cactusToken.isEmpty) {
        debugPrint('⚠️  Running in LOCAL-ONLY mode. Memory-intensive tasks may crash.');
        debugPrint('💡 Add your Cactus token to lib/config.dart to enable hybrid cloud fallback.');
        debugPrint('📍 Get your token from: https://www.cactuscompute.com/dashboard');
      } else {
        debugPrint('✅ Hybrid mode enabled! Heavy compute will use Cactus cloud fallback.');
      }

      setState(() {
        _isInitialized = true;
        _statusMessage = 'Ready';
      });
    } catch (e) {
      setState(() => _statusMessage = 'Initialization failed: $e');
    }
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

    return ProjectsPage(
      conversationService: _conversationService,
      projectService: _projectService!,
    );
  }
}
