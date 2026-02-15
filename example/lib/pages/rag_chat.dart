import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:file_picker/file_picker.dart';

import '../widgets/message_bubble.dart' show AppMessage, MessageBubble;
import '../widgets/input_area.dart' show InputArea;
import '../widgets/document_preview.dart';

import '../services/document_service.dart';
import '../services/conversation_service.dart';
import '../tools/document_similarity_tool.dart';
import 'document_graph_page.dart';

class RAGChatPage extends StatefulWidget {
  final ConversationService conversationService;
  final VoidCallback? onSwitchMode;
  
  const RAGChatPage({
    super.key,
    required this.conversationService,
    this.onSwitchMode,
  });

  @override
  State<RAGChatPage> createState() => _RAGChatPageState();
}

class _RAGChatPageState extends State<RAGChatPage> {
  final _messageController = TextEditingController();
  
  bool _isProcessing = false;
  bool _isAddingDocument = false;
  bool _isSyncingLibrary = false;
  bool _isComputingSimilarity = false;
  
  final List<AppMessage> _messages = [];
  List<Map<String, dynamic>> _pendingDocs = [];
  
  bool get _hasQuery => _messageController.text.trim().isNotEmpty;
  bool get _hasPendingDocs => _pendingDocs.isNotEmpty;

  @override
  void initState() {
    super.initState();
    _messageController.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _messageController.dispose();
    super.dispose();
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

    _addMessage(userQuery, isUser: true);

    try {
      final response = await widget.conversationService.handleQuery(
        query: userQuery,
        newDocs: docsToProcess.isNotEmpty ? docsToProcess : null,
      );
      
      if (docsToProcess.isNotEmpty) {
        setState(() => _pendingDocs.clear());
      }
      
      _addMessage(response, isUser: false);
    } catch (e) {
      _addMessage('Error: $e', isUser: false);
    } finally {
      setState(() => _isProcessing = false);
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

  Future<void> _computeAndShowGraph() async {
    setState(() => _isComputingSimilarity = true);

    try {
      print('🔍 Starting document similarity computation...');
      final result = await DocumentSimilarityTool.execute(
        {'threshold': 0.8},
        widget.conversationService.ragService,
      );
      
      print('🔍 Similarity result: $result');
      print('🔍 Graph data: ${DocumentSimilarityTool.lastComputedGraph}');
      
      if (DocumentSimilarityTool.lastComputedGraph != null) {
        final graph = DocumentSimilarityTool.lastComputedGraph!;
        print('🔍 Nodes: ${graph.nodes.length}, Edges: ${graph.edges.length}');
        
        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => DocumentGraphPage(
                graphData: graph,
              ),
            ),
          );
        }
      } else if (mounted) {
        _showSnackBar('No documents to compare. Upload documents first.');
      }
    } catch (e) {
      print('🔍 Error: $e');
      if (mounted) {
        _showSnackBar('Error computing similarity: $e');
      }
    } finally {
      setState(() => _isComputingSimilarity = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('RAG Chat'),
        actions: [
          IconButton(
            icon: _isComputingSimilarity
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : const Icon(Icons.hub),
            onPressed: _isComputingSimilarity ? null : _computeAndShowGraph,
            tooltip: 'Compute & View Similarity Graph',
          ),
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
