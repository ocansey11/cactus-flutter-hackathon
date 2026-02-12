import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:file_picker/file_picker.dart';

import '../widgets/message_bubble.dart' show AppMessage, MessageBubble;
import '../widgets/input_area.dart' show InputArea;
import '../widgets/document_preview.dart';

import '../services/message_router.dart';
import '../services/rag_service.dart';
import '../services/chat_service.dart';
import '../services/document_service.dart';
import '../services/conversation_service.dart';

class RAGChatPage extends StatefulWidget {
  const RAGChatPage({super.key});

  @override
  State<RAGChatPage> createState() => _RAGChatPageState();
}

class _RAGChatPageState extends State<RAGChatPage> {
  final _embeddingModel = CactusLM();
  final _chatModel = CactusLM();
  final _rag = CactusRAG();
  
  late RAGService _ragService;
  late ChatService _chatService;
  late MessageRouter _messageRouter;
  late ConversationService _conversationService;
  
  final _messageController = TextEditingController();
  
  bool _isInitializing = false;
  bool _isReady = false;
  bool _isProcessing = false;
  bool _isAddingDocument = false;
  bool _isSyncingLibrary = false;
  String _statusMessage = 'Initializing...';
  
  final List<AppMessage> _messages = [];
  List<Map<String, dynamic>> _pendingDocs = [];
  
  bool get _hasQuery => _messageController.text.trim().isNotEmpty;
  bool get _hasPendingDocs => _pendingDocs.isNotEmpty;

  @override
  void initState() {
    super.initState();
    CactusTelemetry.setTelemetryToken('a83c7f7a-43ad-4823-b012-cbeb587ae788');
    _messageController.addListener(() => setState(() {}));
    _initializeSystem();
  }

  @override
  void dispose() {
    _embeddingModel.unload();
    _chatModel.unload();
    _rag.close();
    _messageController.dispose();
    super.dispose();
  }

  Future<void> _initializeSystem() async {
    setState(() {
      _isInitializing = true;
      _statusMessage = 'Downloading FunctionGemma...';
    });

    try {
      await _embeddingModel.downloadModel(
        model: 'functiongemma-270m',
        downloadProcessCallback: (progress, status, isError) {
          setState(() => _statusMessage = isError ? 'Error: $status' : status);
        },
      );
      
      await _embeddingModel.initializeModel(
        params: CactusInitParams(model: 'functiongemma-270m'),
      );

      setState(() => _statusMessage = 'Initializing chat model...');
      await _chatModel.downloadModel(
        model: 'functiongemma-270m',
        downloadProcessCallback: (progress, status, isError) {
          setState(() => _statusMessage = isError ? 'Error: $status' : status);
        },
      );
      
      await _chatModel.initializeModel(
        params: CactusInitParams(model: 'functiongemma-270m'),
      );

      setState(() => _statusMessage = 'Setting up services...');
      _ragService = RAGService(
        rag: _rag,
        embeddingModel: _embeddingModel,
      );
      await _ragService.initialize();
      
      _chatService = ChatService(chatModel: _chatModel);
      _messageRouter = MessageRouter(rag: _rag);
      _conversationService = ConversationService(
        ragService: _ragService,
        chatService: _chatService,
      );

      setState(() {
        _isInitializing = false;
        _isReady = true;
        _statusMessage = 'System ready!';
      });
    } catch (e) {
      setState(() {
        _isInitializing = false;
        _statusMessage = 'Initialization failed: $e';
      });
    }
  }

  Future<void> _addDocument() async {
    try {
      setState(() => _isAddingDocument = true);

      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['txt', 'md', 'pdf'],
      );

      if (result == null || result.files.single.path == null) {
        setState(() => _isAddingDocument = false);
        return;
      }

      final file = result.files.single;
      final extension = file.name.split('.').last.toLowerCase();
      
      final content = await DocumentService.extractContent(
        file.path!,
        extension,
      );

      if (!DocumentService.isValidContent(content)) {
        if (mounted) {
          _showSnackBar('File is empty or could not be read');
        }
        return;
      }

      setState(() {
        _pendingDocs.add(DocumentService.createDocumentMetadata(
          fileName: file.name,
          filePath: file.path!,
          content: content,
          fileSize: file.size,
        ));
      });

      if (mounted) {
        _showSnackBar('Added: ${file.name}');
      }
    } catch (e) {
      if (mounted) {
        _showSnackBar('Error adding document: $e');
      }
    } finally {
      setState(() => _isAddingDocument = false);
    }
  }

  void _removePendingDoc(int index) {
    setState(() => _pendingDocs.removeAt(index));
  }

  Future<void> _selectDocumentLibrary() async {
    try {
      setState(() => _isSyncingLibrary = true);

      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['txt', 'md', 'pdf'],
        allowMultiple: true,
      );

      if (result == null || result.files.isEmpty) {
        setState(() => _isSyncingLibrary = false);
        return;
      }

      final existingDocs = await _ragService.getAllDocuments();
      final existingFileNames = existingDocs.map((d) => d.fileName).toSet();

      final files = result.files
          .where((f) => f.path != null)
          .map((f) => FileInfo(
                fileName: f.name,
                filePath: f.path!,
                extension: f.name.split('.').last.toLowerCase(),
                fileSize: f.size,
              ))
          .toList();

      final importResult = await _conversationService.bulkImport(
        files: files,
        existingFileNames: existingFileNames,
      );

      if (mounted) {
        _showSnackBar(
          'Imported: ${importResult.addedCount} new, ${importResult.skippedCount} skipped',
          duration: const Duration(seconds: 3),
        );
      }
    } catch (e) {
      if (mounted) {
        _showSnackBar('Error importing: $e');
      }
    } finally {
      setState(() => _isSyncingLibrary = false);
    }
  }

  Future<void> _sendMessage() async {
    if (!_hasQuery) {
      if (_hasPendingDocs) {
        _showSnackBar('Document uploaded. Type a question to ask about it.');
      }
      return;
    }

    final userQuery = _messageController.text.trim();
    final docsToProcess = List<Map<String, dynamic>>.from(_pendingDocs);

    _messageController.clear();
    setState(() => _isProcessing = true);

    try {
      final routerResult = await _messageRouter.route(
        query: userQuery,
        hasPendingDocs: docsToProcess.isNotEmpty,
      );

      switch (routerResult.messageType) {
        case MessageType.rag:
          setState(() => _pendingDocs.clear());
          await _handleRAGQuery(userQuery, docsToProcess);
          break;

        case MessageType.simpleChat:
          await _handleSimpleChat(userQuery);
          break;

        case MessageType.toolCalling:
          _addMessage('Tool calling not yet implemented', isUser: false);
          break;
      }
    } catch (e) {
      _addMessage('Error: $e', isUser: false);
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _handleRAGQuery(
    String query,
    List<Map<String, dynamic>> docs,
  ) async {
    _addMessage(query, isUser: true);

    try {
      final response = await _conversationService.handleRAGQuery(
        query: query,
        newDocs: docs,
      );
      _addMessage(response, isUser: false);
    } catch (e) {
      _addMessage('Error: $e', isUser: false);
    }
  }

  Future<void> _handleSimpleChat(String query) async {
    _addMessage(query, isUser: true);

    try {
      final response = await _conversationService.handleSimpleChat(query: query);
      _addMessage(response, isUser: false);
    } catch (e) {
      _addMessage('Error: $e', isUser: false);
    }
  }

  void _addMessage(String text, {required bool isUser}) {
    setState(() {
      _messages.add(AppMessage(text: text, isUser: isUser));
    });
  }

  void _showSnackBar(String message, {Duration? duration}) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          duration: duration ?? const Duration(seconds: 2),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mobile RAG'),
        actions: [
          if (_isReady) ...[
            if (_isSyncingLibrary)
              const Padding(
                padding: EdgeInsets.only(right: 8),
                child: Center(
                  child: SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  ),
                ),
              ),
            PopupMenuButton<String>(
              icon: const Icon(Icons.more_vert),
              onSelected: (value) {
                switch (value) {
                  case 'select_library':
                    _selectDocumentLibrary();
                    break;
                  case 'add_single':
                    _addDocument();
                    break;
                }
              },
              itemBuilder: (context) => [
                const PopupMenuItem(
                  value: 'select_library',
                  child: Row(
                    children: [
                      Icon(Icons.upload_file, size: 20),
                      SizedBox(width: 12),
                      Text('Bulk Import Documents'),
                    ],
                  ),
                ),
                const PopupMenuDivider(),
                PopupMenuItem(
                  value: 'add_single',
                  enabled: !_isAddingDocument,
                  child: Row(
                    children: [
                      Icon(Icons.add, size: 20),
                      SizedBox(width: 12),
                      Text(_isAddingDocument ? 'Adding...' : 'Add Single Doc'),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
      body: _isInitializing
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 20),
                  Text(
                    _statusMessage,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 14),
                  ),
                ],
              ),
            )
          : Column(
              children: [
                Expanded(
                  child: _messages.isEmpty
                      ? Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.chat_bubble_outline,
                                size: 64,
                                color: Colors.grey[300],
                              ),
                              const SizedBox(height: 16),
                              const Text(
                                'Upload a document to start',
                                style: TextStyle(
                                  fontSize: 18,
                                  color: Colors.grey,
                                ),
                              ),
                            ],
                          ),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.all(16),
                          itemCount: _messages.length,
                          itemBuilder: (context, index) {
                            return MessageBubble(message: _messages[index]);
                          },
                        ),
                ),
                DocumentPreview(
                  pendingDocs: _pendingDocs,
                  onRemove: _removePendingDoc,
                ),
                InputArea(
                  isAddingDocument: _isAddingDocument,
                  isProcessing: _isProcessing,
                  onAddDocument: _addDocument,
                  onSend: _sendMessage,
                  messageController: _messageController,
                  canSend: _hasQuery,
                ),
              ],
            ),
    );
  }
}
