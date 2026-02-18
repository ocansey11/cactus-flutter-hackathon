import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:file_picker/file_picker.dart';

import '../widgets/message_bubble.dart' show AppMessage, MessageBubble;
import '../widgets/input_area.dart' show InputArea;
import '../widgets/document_preview.dart';

import '../services/document_service.dart';
import '../services/conversation_service.dart';
import '../services/project_service.dart';
import '../tools/document_similarity_tool.dart';
import 'document_graph_page.dart';
import 'papers_page.dart';
import 'notes_page.dart';
import 'voice_chat.dart';

class RAGChatPage extends StatefulWidget {
  final ConversationService conversationService;
  final ProjectService? projectService;
  final String? currentConversationId;
  final VoidCallback? onSwitchMode;
  final VoidCallback? onOpenDrawer;

  const RAGChatPage({
    super.key,
    required this.conversationService,
    this.projectService,
    this.currentConversationId,
    this.onSwitchMode,
    this.onOpenDrawer,
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

  List<AppMessage> _messages = [];
  List<Map<String, dynamic>> _pendingDocs = [];

  bool get _hasQuery => _messageController.text.trim().isNotEmpty;
  bool get _hasPendingDocs => _pendingDocs.isNotEmpty;

  Project? get _currentProject =>
      widget.projectService?.currentProject;

  @override
  void initState() {
    super.initState();
    _messageController.addListener(() => setState(() {}));
    _loadMessages();
  }

  void _loadMessages() {
    final convoId = widget.currentConversationId;
    if (convoId == null || widget.projectService == null) return;
    final stored = widget.projectService!.getMessages(convoId);
    setState(() {
      _messages = stored
          .map((m) => AppMessage(text: m.text, isUser: m.isUser))
          .toList();
    });
  }

  @override
  void didUpdateWidget(RAGChatPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.currentConversationId != widget.currentConversationId) {
      _loadMessages();
    }
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

      if (result == null || result.files.single.path == null) return;

      final file = result.files.single;
      final content = await DocumentService.extractContent(
        file.path!,
        file.name.split('.').last.toLowerCase(),
      );

      if (!DocumentService.isValidContent(content)) {
        _showSnackBar('File is empty or could not be read');
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

      _showSnackBar('Added: ${file.name}');
    } catch (e) {
      _showSnackBar('Error adding document: $e');
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

      if (result == null || result.files.isEmpty) return;

      final existingDocs =
          await widget.conversationService.ragService!.getAllDocuments();
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
        projectName: _currentProject?.name,
      );

      // Register newly imported docs in DocumentMetadataStore
      if (_currentProject != null && importResult.addedCount > 0) {
        for (final file in files) {
          if (!existingFileNames.contains(file.fileName)) {
            widget.projectService!.registerDocument(
              projectId: _currentProject!.id,
              fileName: file.fileName,
              filePath: file.filePath,
              fileSize: file.fileSize,
            );
          }
        }
      }

      _showSnackBar(
        'Imported: ${importResult.addedCount} new, ${importResult.skippedCount} skipped',
        duration: const Duration(seconds: 3),
      );
    } catch (e) {
      _showSnackBar('Error importing: $e');
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
        projectName: _currentProject?.name,
      );

      // Register newly stored docs in DocumentMetadataStore
      if (docsToProcess.isNotEmpty && _currentProject != null) {
        for (final doc in docsToProcess) {
          widget.projectService!.registerDocument(
            projectId: _currentProject!.id,
            fileName: doc['fileName'],
            filePath: doc['filePath'] ?? '',
            fileSize: doc['fileSize'] ?? 0,
          );
        }
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
    setState(() => _messages.add(AppMessage(text: text, isUser: isUser)));
    // Persist to ObjectBox
    final convoId = widget.currentConversationId;
    if (convoId != null && widget.projectService != null) {
      widget.projectService!.addMessage(
        conversationId: convoId,
        text: text,
        isUser: isUser,
      );
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

  Future<void> _computeAndShowGraph() async {
    setState(() => _isComputingSimilarity = true);
    try {
      final tool = DocumentSimilarityTool();
      await tool.call(
        {'threshold': 0.8},
        ragService: widget.conversationService.ragService,
        projectService: widget.projectService,
      );

      final graph = DocumentGraphStore.lastGraph;
      if (graph != null && mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => DocumentGraphPage(graphData: graph),
          ),
        );
      } else if (mounted) {
        _showSnackBar('No documents to compare. Upload documents first.');
      }
    } catch (e) {
      _showSnackBar('Error computing similarity: $e');
    } finally {
      setState(() => _isComputingSimilarity = false);
    }
  }

  Future<void> _navigateToPapers() async {
    if (widget.projectService == null) return;
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PapersPage(
          projectService: widget.projectService!,
          ragService: widget.conversationService.ragService,
        ),
      ),
    );
    if (mounted) setState(() {});
  }

  void _openVoiceChat() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => VoiceChatPage(
          conversationService: widget.conversationService,
          onSwitchMode: () => Navigator.pop(context),
        ),
      ),
    );
  }

  Future<void> _navigateToNotes() async {
    if (widget.projectService == null) return;
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) =>
            NotesPage(projectService: widget.projectService!),
      ),
    );
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.menu),
          onPressed: widget.onOpenDrawer,
        ),
        title: Text(_currentProject?.name ?? 'Research Assistant'),
        actions: [
          IconButton(
            icon: const Icon(Icons.mic),
            onPressed: widget.onSwitchMode,
            tooltip: 'Switch to Voice',
          ),
          IconButton(
            icon: _isComputingSimilarity
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor:
                          AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : const Icon(Icons.hub),
            onPressed:
                _isComputingSimilarity ? null : _computeAndShowGraph,
            tooltip: 'Document Similarity Graph',
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
                    valueColor:
                        AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
              ),
            ),
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert),
            onSelected: (value) {
              if (value == 'select_library') _selectDocumentLibrary();
              if (value == 'add_single') _addDocument();
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
                    const Icon(Icons.add, size: 20),
                    const SizedBox(width: 12),
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
          if (_currentProject != null)
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Theme.of(context)
                    .colorScheme
                    .surfaceVariant
                    .withOpacity(0.3),
                border: Border(
                  bottom: BorderSide(
                    color: Theme.of(context).dividerColor,
                    width: 1,
                  ),
                ),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextButton.icon(
                      onPressed: _navigateToPapers,
                      icon: const Icon(Icons.description_outlined, size: 20),
                      label: const Text('Papers'),
                      style: TextButton.styleFrom(
                          alignment: Alignment.centerLeft),
                    ),
                  ),
                  Container(
                      width: 1,
                      height: 24,
                      color: Theme.of(context).dividerColor),
                  Expanded(
                    child: TextButton.icon(
                      onPressed: _navigateToNotes,
                      icon: const Icon(Icons.note_outlined, size: 20),
                      label: const Text('Notes'),
                      style: TextButton.styleFrom(
                          alignment: Alignment.centerLeft),
                    ),
                  ),
                ],
              ),
            ),
          Expanded(
            child: _messages.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.chat_bubble_outline,
                            size: 64, color: Colors.grey[300]),
                        const SizedBox(height: 16),
                        const Text(
                          'Upload a document to start',
                          style:
                              TextStyle(fontSize: 18, color: Colors.grey),
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) =>
                        MessageBubble(message: _messages[index]),
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
