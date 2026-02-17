import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:file_picker/file_picker.dart';

import '../widgets/document_preview.dart';
import '../widgets/input_area.dart';

import '../services/message_router.dart';
import '../services/document_service.dart';
import '../services/conversation_service.dart';
import 'package:cactus/memory/conversation_store.dart';


class RAGChatPage extends StatefulWidget {
  final ConversationService conversationService;
  final ConversationStore conversationStore;
  final String currentConversationId;
  final VoidCallback? onSwitchMode;
  final VoidCallback? onOpenDrawer;
  
  const RAGChatPage({
    super.key,
    required this.conversationService,
    required this.conversationStore,
    required this.currentConversationId,
    this.onSwitchMode,
    this.onOpenDrawer,
  });

  @override
  State<RAGChatPage> createState() => _RAGChatPageState();
}

class _RAGChatPageState extends State<RAGChatPage> {
  late MessageRouter _messageRouter;
  
  final _messageController = TextEditingController();
  
  bool _isProcessing = false;
  bool _isAddingDocument = false;
  bool _isSyncingLibrary = false;
  
  List<Message> _messages = [];
  List<Map<String, dynamic>> _pendingDocs = [];
  
  bool get _hasQuery => _messageController.text.trim().isNotEmpty;
  bool get _hasPendingDocs => _pendingDocs.isNotEmpty;

  @override
  void initState() {
    super.initState();
    _loadMessages();
    _messageController.addListener(() => setState(() {}));
    _messageRouter = MessageRouter(
      rag: widget.conversationService.ragService!.rag,
      functionService: widget.conversationService.ragService != null 
          ? MessageRouter(
              rag: widget.conversationService.ragService!.rag,
            ).functionService
          : null,
    );
  }

  @override
  void dispose() {
    _messageController.dispose();
    super.dispose();
  }

  void _loadMessages() {
    setState(() {
      _messages = widget.conversationStore.getMessages(widget.currentConversationId);
    });
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

      final existingDocs = await widget.conversationService.ragService!.getAllDocuments();
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

      final importResult = await widget.conversationService.bulkImport(
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
      await widget.conversationStore.addMessage(
        conversationId: widget.currentConversationId,
        text: userQuery,
        isUser: true,
      );
      _loadMessages();

      final routeResult = await _messageRouter.route(
        query: userQuery,
        hasPendingDocs: docsToProcess.isNotEmpty,
      );

      String response;
      switch (routeResult.messageType) {
        case MessageType.rag:
          setState(() => _pendingDocs.clear());
          response = await widget.conversationService.handleQuery(
            query: userQuery,
            newDocs: docsToProcess,
          );
          break;

        case MessageType.simpleChat:
          response = await widget.conversationService.handleQuery(
            query: userQuery,
            newDocs: [],
          );
          break;

        case MessageType.toolCalling:
          response = await widget.conversationService.handleSimpleChat(query: userQuery);
          break;
      }

      await widget.conversationStore.addMessage(
        conversationId: widget.currentConversationId,
        text: response,
        isUser: false,
      );
      _loadMessages();
    } catch (e) {
      await widget.conversationStore.addMessage(
        conversationId: widget.currentConversationId,
        text: 'Error: $e',
        isUser: false,
      );
      _loadMessages();
    } finally {
      setState(() => _isProcessing = false);
    }
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
        leading: IconButton(
          icon: const Icon(Icons.menu),
          onPressed: widget.onOpenDrawer,
        ),
        title: const Text('RAG Chat'),
        actions: [
          IconButton(
            icon: const Icon(Icons.mic),
            onPressed: widget.onSwitchMode,
            tooltip: 'Switch to Voice Chat',
          ),
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
      ),
      body: Column(
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
                      final message = _messages[index];
                      return _buildMessageBubble(message);
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

  Widget _buildMessageBubble(Message message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        decoration: BoxDecoration(
          color: message.isUser ? Colors.blue : Colors.grey[200],
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          message.text,
          style: TextStyle(
            color: message.isUser ? Colors.white : Colors.black,
            fontSize: 15,
          ),
        ),
      ),
    );
  }
}