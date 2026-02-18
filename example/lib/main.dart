import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'pages/rag_chat.dart';
import 'pages/voice_chat.dart';
import 'services/conversation_service.dart';
import 'services/project_service.dart';
import 'services/chat_service.dart';
import 'services/rag_service.dart';
import 'services/function_calling_service.dart';
import 'services/message_router.dart';
import 'services/model_manager.dart';
import 'widgets/app_drawer.dart';

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
  bool _isVoiceMode = false;
  String _statusMessage = 'Initializing...';
  double? _loadProgress;
  final List<String> _logLines = [];
  final _scaffoldKey = GlobalKey<ScaffoldState>();

  // Current conversation state
  String? _currentConversationId;

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

  void _log(String line) {
    setState(() {
      _logLines.add(line);
      _statusMessage = line.replaceAll(RegExp(r'^[✓✗⟳] '), '');
    });
  }

  Future<void> _initializeServices() async {
    try {
      // ── Embedding model (Qwen) ──────────────────────────────────────────
      _log('⟳ Downloading embedding model (qwen3-0.6)...');
      _embeddingModel = await ModelManager.getOrInitializeLLM(
        modelName: 'qwen3-0.6',
        progressCallback: (progress, status, isError) {
          setState(() {
            _loadProgress = progress;
            _statusMessage = isError ? '✗ $status' : status;
          });
          if (isError) _logLines.add('✗ $status');
        },
      );
      _log('✓ Embedding model ready');

      // ── Function dispatch model (Gemma) ─────────────────────────────────
      _log('⟳ Downloading function model (gemma3-270m)...');
      setState(() => _loadProgress = null);
      _functionModel = await ModelManager.getOrInitializeLLM(
        modelName: 'gemma3-270m',
        progressCallback: (progress, status, isError) {
          setState(() {
            _loadProgress = progress;
            _statusMessage = isError ? '✗ $status' : status;
          });
          if (isError) _logLines.add('✗ $status');
        },
      );
      _log('✓ Function model ready');

      // ── Chat model (Qwen, shared instance) ──────────────────────────────
      _log('⟳ Initializing chat model (qwen3-0.6)...');
      setState(() => _loadProgress = null);
      _chatModel = await ModelManager.getOrInitializeLLM(
        modelName: 'qwen3-0.6',
        progressCallback: (progress, status, isError) {
          setState(() {
            _loadProgress = progress;
            _statusMessage = isError ? '✗ $status' : status;
          });
          if (isError) _logLines.add('✗ $status');
        },
      );
      _log('✓ Chat model ready');

      // ── Services ────────────────────────────────────────────────────────
      _log('⟳ Initializing RAG service...');
      final ragService = RAGService(rag: _rag, embeddingModel: _embeddingModel);
      await ragService.initialize();
      _log('✓ RAG service ready');

      _log('⟳ Initializing ObjectBox...');
      await ObjectBoxManager.initialize(_rag);
      _log('✓ ObjectBox ready');

      _log('⟳ Wiring project services...');
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
      _log('✓ All services ready');

      // Auto-select first project + conversation if any exist
      final projects = _projectService.getAllProjects();
      if (projects.isNotEmpty) {
        _projectService.setCurrentProject(projects.first);
        final convos = _projectService.getConversations(projectId: projects.first.id);
        if (convos.isNotEmpty) {
          _currentConversationId = convos.first.id;
        } else {
          final convo = await _projectService.createConversation(
            projectId: projects.first.id,
            title: 'Chat 1',
          );
          _currentConversationId = convo.id;
        }
      }

      setState(() {
        _isInitialized = true;
        _loadProgress = 1.0;
        _statusMessage = 'Ready';
      });
    } catch (e) {
      _log('✗ Initialization failed: $e');
    }
  }

  void _onSelectConversation(Project project, Conversation convo) {
    setState(() {
      _currentConversationId = convo.id;
      _isVoiceMode = false;
    });
  }

  Future<void> _onNewConversation(Project project) async {
    final convos = _projectService.getConversations(projectId: project.id);
    final convo = await _projectService.createConversation(
      projectId: project.id,
      title: 'Chat ${convos.length + 1}',
    );
    setState(() {
      _currentConversationId = convo.id;
      _isVoiceMode = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Logo / title
                Row(
                  children: [
                    Icon(Icons.memory,
                        size: 32,
                        color: Theme.of(context).colorScheme.primary),
                    const SizedBox(width: 12),
                    Text(
                      'JarvisOS',
                      style: Theme.of(context)
                          .textTheme
                          .headlineSmall
                          ?.copyWith(fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  'On-device AI — loading models',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Theme.of(context)
                            .colorScheme
                            .onSurface
                            .withOpacity(0.6),
                      ),
                ),
                const SizedBox(height: 48),

                // Progress bar
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: _loadProgress,
                    minHeight: 6,
                    backgroundColor: Theme.of(context)
                        .colorScheme
                        .surfaceContainerHighest,
                  ),
                ),
                const SizedBox(height: 16),

                // Current step
                Text(
                  _statusMessage,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w500,
                      ),
                ),
                const SizedBox(height: 4),

                // Percentage
                if (_loadProgress != null)
                  Text(
                    '${(_loadProgress! * 100).toStringAsFixed(1)}%',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Theme.of(context)
                              .colorScheme
                              .onSurface
                              .withOpacity(0.5),
                          fontFeatures: const [FontFeature.tabularFigures()],
                        ),
                  ),

                const SizedBox(height: 32),

                // Step log
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Theme.of(context)
                        .colorScheme
                        .surfaceContainerHighest
                        .withOpacity(0.4),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Initialization log',
                        style: Theme.of(context)
                            .textTheme
                            .labelSmall
                            ?.copyWith(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 8),
                      ..._logLines.reversed.take(6).toList().reversed.map(
                            (line) => Padding(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 1),
                              child: Text(
                                line,
                                style: Theme.of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(
                                      fontFamily: 'monospace',
                                      color: line.startsWith('✓')
                                          ? Colors.green[700]
                                          : line.startsWith('✗')
                                              ? Colors.red[700]
                                              : null,
                                    ),
                              ),
                            ),
                          ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // No project selected — show a welcome/empty state
    if (_projectService.currentProject == null || _currentConversationId == null) {
      return Scaffold(
        key: _scaffoldKey,
        drawer: AppDrawer(
          projectService: _projectService,
          currentConversationId: _currentConversationId,
          onSelectConversation: _onSelectConversation,
          onNewConversation: _onNewConversation,
        ),
        appBar: AppBar(
          leading: IconButton(
            icon: const Icon(Icons.menu),
            onPressed: () => _scaffoldKey.currentState?.openDrawer(),
          ),
          title: const Text('JarvisOS'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.folder_open, size: 80, color: Colors.grey[300]),
              const SizedBox(height: 16),
              Text('No project selected',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(color: Colors.grey)),
              const SizedBox(height: 8),
              Text('Open the sidebar to create or select a project',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey)),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: () => _scaffoldKey.currentState?.openDrawer(),
                icon: const Icon(Icons.menu),
                label: const Text('Open Sidebar'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      key: _scaffoldKey,
      drawer: AppDrawer(
        projectService: _projectService,
        currentConversationId: _currentConversationId,
        onSelectConversation: _onSelectConversation,
        onNewConversation: _onNewConversation,
      ),
      body: _isVoiceMode
          ? VoiceChatPage(
              conversationService: _conversationService,
              onSwitchMode: () => setState(() => _isVoiceMode = false),
              onOpenDrawer: () => _scaffoldKey.currentState?.openDrawer(),
            )
          : RAGChatPage(
              conversationService: _conversationService,
              projectService: _projectService,
              currentConversationId: _currentConversationId!,
              onSwitchMode: () => setState(() => _isVoiceMode = true),
              onOpenDrawer: () => _scaffoldKey.currentState?.openDrawer(),
            ),
    );
  }
}
