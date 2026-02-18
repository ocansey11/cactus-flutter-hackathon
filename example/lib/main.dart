import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'pages/projects_page.dart';
import 'services/conversation_service.dart';
import 'services/project_service.dart';
import 'services/chat_service.dart';
import 'services/rag_service.dart';
import 'services/function_calling_service.dart';
import 'services/message_router.dart';
import 'services/model_manager.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus',
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
  late CactusLM _functionModel;
  late ConversationService _conversationService;
  late ProjectService _projectService;

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
      setState(() => _statusMessage = 'Loading embedding model...');
      _embeddingModel = await ModelManager.getOrInitializeLLM(
        modelName: 'qwen3-0.6',
        progressCallback: (progress, status, isError) {
          setState(() => _statusMessage = isError ? 'Error: $status' : status);
        },
      );

      setState(() => _statusMessage = 'Loading function model...');
      _functionModel = await ModelManager.getOrInitializeLLM(
        modelName: 'gemma3-270m',
        progressCallback: (progress, status, isError) {
          setState(() => _statusMessage = isError ? 'Error: $status' : status);
        },
      );

      setState(() => _statusMessage = 'Loading chat model...');
      _chatModel = await ModelManager.getOrInitializeLLM(
        modelName: 'qwen3-0.6',
        progressCallback: (progress, status, isError) {
          setState(() => _statusMessage = isError ? 'Error: $status' : status);
        },
      );

      setState(() => _statusMessage = 'Setting up services...');

      final ragService = RAGService(rag: _rag, embeddingModel: _embeddingModel);
      await ragService.initialize();

      await ObjectBoxManager.initialize(_rag);

      final projectStore = ProjectStore();
      final conversationStore = ConversationStore(embeddingModel: _embeddingModel);
      final metadataStore = DocumentMetadataStore();
      final noteStore = NoteStore();

      _projectService = ProjectService(
        projectStore: projectStore,
        conversationStore: conversationStore,
        metadataStore: metadataStore,
        noteStore: noteStore,
      );

      final chatService = ChatService(chatModel: _chatModel);

      final functionService = FunctionCallingService(
        queryAnalyzer: _chatModel,
        functionModel: _functionModel,
      );

      final messageRouter = MessageRouter(
        rag: _rag,
        functionService: functionService,
      );

      _conversationService = ConversationService(
        chatService: chatService,
        ragService: ragService,
        messageRouter: messageRouter,
        projectService: _projectService,
      );

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
      projectService: _projectService,
    );
  }
}
