import 'package:flutter/material.dart';
import 'dart:io';
import 'package:cactus/cactus.dart';
import 'package:file_picker/file_picker.dart';
import 'package:read_pdf_text/read_pdf_text.dart';

// Import our custom widgets
import '../widgets/message_bubble.dart' show AppMessage, MessageBubble;
import '../widgets/input_area.dart' show InputArea;
import '../widgets/document_preview.dart';

/// Main RAG Chat Page
class RAGChatPage extends StatefulWidget {
  const RAGChatPage({super.key});

  @override
  State<RAGChatPage> createState() => _RAGChatPageState();
}

class _RAGChatPageState extends State<RAGChatPage> {
  // Cactus components
  final _embeddingModel = CactusLM();
  final _chatModel = CactusLM();
  final _rag = CactusRAG();

  // Controllers
  final _messageController = TextEditingController();

  /// Strip thinking process tags and special tokens from model output
  String _stripThinkingTags(String text) {
    // Remove content within <think>...</think> tags (case-insensitive)
    final regex =
        RegExp(r'<think>.*?</think>', caseSensitive: false, dotAll: true);
    var cleaned = text.replaceAll(regex, '');

    // Remove common end tokens that might leak through
    cleaned = cleaned.replaceAll('<|im_end|>', '');
    cleaned = cleaned.replaceAll('<end_of_turn>', '');
    cleaned = cleaned.replaceAll('<|endoftext|>', '');

    return cleaned.trim();
  }

  // State flags
  bool _isInitializing = false;
  bool _isReady = false;
  bool _isProcessing = false;

  // Data storage
  final List<AppMessage> _messages = [];

  // Document state
  bool _isAddingDocument = false;
  List<Map<String, dynamic>> _pendingDocs = [];

  // Track if we're using RAG mode or describe mode
  bool get _hasQuery => _messageController.text.trim().isNotEmpty;
  bool get _hasPendingDocs => _pendingDocs.isNotEmpty;

  // Status tracking
  String _statusMessage = 'Initializing...';

  // Bulk import state
  bool _isSyncingLibrary = false;

  @override
  void initState() {
    super.initState();
    CactusTelemetry.setTelemetryToken('a83c7f7a-43ad-4823-b012-cbeb587ae788');
    // Add listener to rebuild UI when text changes (for canSend state)
    _messageController.addListener(() {
      setState(() {});
    });
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

  /// Initialize the entire RAG system
  Future<void> _initializeSystem() async {
    setState(() {
      _isInitializing = true;
      _statusMessage = 'Downloading embedding model...';
    });

    try {
      // Step 1: Download and initialize embedding model
      await _embeddingModel.downloadModel(
        model: 'qwen3-0.6-embed',
        downloadProcessCallback: (progress, status, isError) {
          setState(() {
            _statusMessage = isError ? 'Error: $status' : status;
          });
        },
      );

      setState(() => _statusMessage = 'Initializing embedding model...');
      await _embeddingModel.initializeModel(
        params: CactusInitParams(model: 'qwen3-0.6-embed'),
      );

      // Step 2: Download and initialize chat model (Qwen 3 - 600M)
      setState(() => _statusMessage = 'Downloading chat model...');
      await _chatModel.downloadModel(
        model: 'qwen3-0.6',
        downloadProcessCallback: (progress, status, isError) {
          setState(() {
            _statusMessage = isError ? 'Error: $status' : status;
          });
        },
      );

      setState(() => _statusMessage = 'Initializing chat model...');
      await _chatModel.initializeModel(
        params: CactusInitParams(model: 'qwen3-0.6'),
      );

      // Step 3: Initialize RAG database
      setState(() => _statusMessage = 'Setting up vector database...');
      await _rag.initialize();
      _rag.setEmbeddingGenerator((text) async {
        final result = await _embeddingModel.generateEmbedding(text: text);
        return result.embeddings;
      });
      _rag.setChunking(chunkSize: 1024, chunkOverlap: 128);

      // Step 4: Mark as ready
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

  /// Add a document (txt, md, or pdf)
  Future<void> _addDocument() async {
    try {
      setState(() => _isAddingDocument = true);
      // Pending documents (uploaded but not yet described)

      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['txt', 'md', 'pdf'],
      );

      if (result == null || result.files.single.path == null) {
        setState(() => _isAddingDocument = false);
        return;
      }

      final filePath = result.files.single.path!;
      final fileName = result.files.single.name;
      final fileSize = result.files.single.size;
      final extension = fileName.split('.').last.toLowerCase();

      String content;

      // Extract text based on file type
      if (extension == 'pdf') {
        content = await ReadPdfText.getPDFtext(filePath);

        // Fix common PDF extraction corruptions
        content =
            content.replaceAll('"', 'a'); // Replace corrupted 'a' characters
        content = content.replaceAll('ʼ', "'"); // Fix apostrophes
        print('PDF text extracted and cleaned (${content.length} chars)');
      } else {
        // For .txt and .md files
        content = await File(filePath).readAsString();
      }

      if (content.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('File is empty or could not be read')),
          );
        }
        return;
      }

      // // Store document in RAG database - Might need to be done when sent button is clicked
      // await _rag.storeDocument(
      //   fileName: fileName,
      //   filePath: filePath,
      //   content: content,
      //   fileSize: fileSize,
      // );

      // Add to pending docs list (in-chat attachment)
      setState(() {
        _pendingDocs.add({
          'fileName': fileName,
          'fileSize': fileSize,
          'content': content,
        });
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Added: $fileName')),
        );
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Added: $fileName')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error adding document: $e')),
        );
      }
    } finally {
      setState(() => _isAddingDocument = false);
    }
  }

  /// Remove a pending document
  void _removePendingDoc(int index) {
    setState(() {
      _pendingDocs.removeAt(index);
    });
  }

  /// Select multiple documents to add to library at once
  Future<void> _selectDocumentLibrary() async {
    try {
      setState(() => _isSyncingLibrary = true);

      // On iOS, directory access is temporary, so use multi-file picker instead
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['txt', 'md', 'pdf'],
        allowMultiple: true, // Select multiple files at once
      );

      if (result == null || result.files.isEmpty) {
        setState(() => _isSyncingLibrary = false);
        return; // User cancelled
      }

      print('=== BULK IMPORT: ${result.files.length} files selected ===');

      // Get existing documents in database
      final existingDocs = await _rag.getAllDocuments();
      final existingFileNames = existingDocs.map((d) => d.fileName).toSet();

      int addedCount = 0;
      int skippedCount = 0;

      // Process each selected file
      for (final platformFile in result.files) {
        if (platformFile.path == null) {
          print('Skipping file with no path');
          skippedCount++;
          continue;
        }

        final file = File(platformFile.path!);
        final fileName = platformFile.name;

        // Skip if already in database
        if (existingFileNames.contains(fileName)) {
          print('Skipping (already exists): $fileName');
          skippedCount++;
          continue;
        }

        try {
          print('Processing: $fileName');

          String content;
          final extension = fileName.split('.').last.toLowerCase();

          if (extension == 'pdf') {
            content = await ReadPdfText.getPDFtext(file.path);
            // Fix common PDF extraction corruptions
            content = content.replaceAll('"', 'a');
            content = content.replaceAll('ʼ', "'");
          } else {
            content = await file.readAsString();
          }

          if (content.trim().isEmpty) {
            print('Skipping (empty): $fileName');
            skippedCount++;
            continue;
          }

          // Store in RAG database
          await _rag.storeDocument(
            fileName: fileName,
            filePath: file.path,
            content: content,
            fileSize: platformFile.size,
          );

          addedCount++;
          print('Added: $fileName');
        } catch (e) {
          print('Error processing $fileName: $e');
          skippedCount++;
        }
      }

      print(
          '=== BULK IMPORT COMPLETE: $addedCount added, $skippedCount skipped ===');

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Imported: $addedCount new, $skippedCount skipped',
            ),
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } catch (e) {
      print('Error importing documents: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error importing: $e')),
        );
      }
    } finally {
      setState(() => _isSyncingLibrary = false);
    }
  }

  /// Smart send: RAG search if query exists, auto-describe if only docs
  Future<void> _sendMessage() async {
    // Require a user-typed query. If documents are uploaded but no query
    // is provided, prompt the user to type a question to query the documents.
    if (!_hasQuery) {
      if (_hasPendingDocs) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text(
                'Document uploaded. Type a question to ask about the document(s).',
              ),
            ),
          );
        }
      }
      return;
    }

    final userQuery = _messageController.text.trim();
    final docsToProcess = List<Map<String, dynamic>>.from(_pendingDocs);

    // Clear input and mark processing. Keep pending docs until used below.
    _messageController.clear();
    setState(() {
      _isProcessing = true;
    });

    try {
      // Always use RAG search if database has documents OR new docs are being added
      if (docsToProcess.isNotEmpty) {
        // Store new documents and search
        setState(() => _pendingDocs.clear());
        await _ragSearchWithDocs(userQuery, docsToProcess);
      } else {
        // No new documents, but check if RAG database has existing documents
        final existingDocs = await _rag.getAllDocuments();
        if (existingDocs.isNotEmpty) {
          // Search existing documents in RAG database
          await _ragSearchWithDocs(userQuery, []);
        } else {
          // Database is empty, fall back to simple chat
          await _simpleQuery(userQuery);
        }
      }
    } catch (e) {
      setState(() {
        _messages.add(AppMessage(
          text: 'Error: $e',
          isUser: false,
        ));
      });
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  /// Mode 1: RAG search with documents
  Future<void> _ragSearchWithDocs(
      String query, List<Map<String, dynamic>> docs) async {
    // Add user message
    setState(() {
      _messages.add(AppMessage(text: query, isUser: true));
    });

    try {
      // Store any NEW documents (if provided)
      if (docs.isNotEmpty) {
        print('=== STORING NEW DOCUMENTS ===');
        for (final doc in docs) {
          print('Storing: ${doc['fileName']}');
          final storedDoc = await _rag.storeDocument(
            fileName: doc['fileName'],
            filePath: '',
            content: doc['content'],
            fileSize: doc['fileSize'],
          );
          print(
              'Stored: ${doc['fileName']} with ${storedDoc.chunks.length} chunks');
        }
        print('Successfully stored ${docs.length} new documents');
      }

      // Show all documents in database
      final allDocs = await _rag.getAllDocuments();
      print('=== CURRENT DATABASE ===');
      print('Total documents: ${allDocs.length}');
      for (final doc in allDocs) {
        print('  - ${doc.fileName} (${doc.chunks.length} chunks)');
      }
      print('=== END DATABASE STATE ===');

      // Search RAG database with the user's query to find relevant chunks
      final results = await _rag.search(text: query, limit: 5);

      print('Search returned ${results.length} results for query: "$query"');

      // Debug: print which documents the results came from
      for (final result in results) {
        final docName = result.chunk.document.target?.fileName ?? 'unknown';
        print(
            'Result chunk from document: $docName (distance: ${result.distance})');
      }

      if (results.isEmpty) {
        setState(() {
          _messages.add(AppMessage(
            text:
                'No relevant content found in the uploaded documents. The document may be empty or the embeddings failed.',
            isUser: false,
          ));
        });
        return;
      }

      // Build context from results - limit context size
      final contextChunks = results.map((r) => r.chunk.content).toList();
      var context = contextChunks.join('\n\n---\n\n');

      // Debug: print FULL context to terminal
      print('=== FULL RAG CONTEXT (${context.length} chars) ===');
      print(context);
      print('=== END FULL CONTEXT ===');

      // Verify context is not empty
      if (context.trim().isEmpty) {
        print('ERROR: Context is empty after joining chunks!');
        setState(() {
          _messages.add(AppMessage(
            text:
                'Error: Retrieved chunks are empty. Document may not have been stored properly.',
            isUser: false,
          ));
        });
        return;
      }

      // Truncate if context is too long (keep it under ~2000 chars for the model)
      if (context.length > 2000) {
        context = context.substring(0, 2000) + '\n...[truncated for length]';
        print('Context truncated to 2000 chars');
      }

      // Generate response with context - VERY explicit instructions
      final systemPrompt =
          'You are a document Q&A assistant. You must ONLY use the information provided in the Context section below. DO NOT use your general knowledge. If the answer is not in the Context, say "I cannot find that information in the provided document."';

      final userPrompt = '''Here is the content from the uploaded document:

CONTEXT START:
$context
CONTEXT END:

Now answer this question using ONLY the information above: $query

Remember: Only use information from the CONTEXT section above. Do not add information from your training data.''';

      print('=== SENDING TO MODEL ===');
      print('System: ${systemPrompt.substring(0, 100)}...');
      print('User prompt length: ${userPrompt.length} chars');

      final response = await _chatModel.generateCompletion(
        messages: [
          ChatMessage(content: systemPrompt, role: 'system'),
          ChatMessage(content: userPrompt, role: 'user'),
        ],
        params: CactusCompletionParams(
          maxTokens: 2000,
          temperature: 0.3, // Lower temperature for more factual responses
          stopSequences: [],
        ),
      );

      if (!response.success) {
        setState(() {
          _messages.add(AppMessage(
            text:
                'Failed to generate response. Try with a shorter query or document.',
            isUser: false,
          ));
        });
        return;
      }

      print('=== RAW MODEL RESPONSE ===');
      print(response.response);
      print('=== END RAW RESPONSE ===');

      final cleanedResponse = _stripThinkingTags(response.response);
      print('=== CLEANED RESPONSE ===');
      print(cleanedResponse);
      print('=== END CLEANED ===');

      setState(() {
        _messages.add(AppMessage(
          text: cleanedResponse,
          isUser: false,
        ));
      });
    } catch (e) {
      setState(() {
        _messages.add(AppMessage(
          text: 'Error: $e',
          isUser: false,
        ));
      });
    }
  }

  /// Mode 2: Simple query without documents
  /// Simple query without documents
  Future<void> _simpleQuery(String query) async {
    // Add user message
    setState(() {
      _messages.add(AppMessage(text: query, isUser: true));
    });

    try {
      // Get LLM response
      final response = await _chatModel.generateCompletion(
        messages: [
          ChatMessage(
              content: 'You are Cactus, a helpful AI assistant.',
              role: 'system'),
          ChatMessage(content: query, role: 'user'),
        ],
        params: CactusCompletionParams(
          maxTokens: 2000,
          temperature: 0.7,
          stopSequences: [],
        ),
      );

      if (!response.success) {
        setState(() {
          _messages.add(AppMessage(
            text: 'Failed to generate response. Please try again.',
            isUser: false,
          ));
        });
        return;
      }

      // Add AI response
      setState(() {
        _messages.add(AppMessage(
          text: _stripThinkingTags(response.response),
          isUser: false,
        ));
      });
    } catch (e) {
      setState(() {
        _messages.add(AppMessage(
          text: 'Error: $e',
          isUser: false,
        ));
      });
    }
  }
  
  /// Mode 3: Auto-describe uploaded documents
  Future<void> _autoDescribe(List<Map<String, dynamic>> docs) async {
    // Add user message showing what was uploaded
    final fileNames = docs.map((d) => d['fileName'] as String).join(', ');
    setState(() {
      _messages.add(AppMessage(
        text: 'Uploaded: $fileNames',
        isUser: true,
      ));
    });
    
    try {
      // Combine all document content
      final combinedContent = docs.map((d) => d['content'] as String).join('\n\n---\n\n');
      
      // Create summary prompt
      final prompt = '''I have uploaded ${docs.length} document(s). Please provide a brief summary of what these documents contain (2-3 sentences):

$combinedContent

Summary:''';
      
      // Get LLM response
      final response = await _chatModel.generateCompletion(
        messages: [
          ChatMessage(content: prompt, role: 'user'),
        ],
      );
      
      // Add AI response
      setState(() {
        _messages.add(AppMessage(
          text: response.response,
          isUser: false,
        ));
      });
    } catch (e) {
      setState(() {
        _messages.add(AppMessage(
          text: 'Error describing documents: $e',
          isUser: false,
        ));
      });
    }
  }

  Future<void> _ragSearch(String query, List<Map<String, dynamic>> docs) async {
    // Add user message
    final fileNames = docs.map((d) => d['fileName'] as String).join(', ');
    setState(() {
      _messages.add(AppMessage(
        text: '$query\n\nDocuments: $fileNames',
        isUser: true,
      ));
    });
    
    try {
      // Step 1: Embed and store documents NOW (not during upload)
      for (var doc in docs) {
        await _rag.storeDocument(
          fileName: doc['fileName'],
          filePath: '', // Not needed for search
          content: doc['content'],
          fileSize: doc['fileSize'],
        );
      }
      
      // Step 2: Search for relevant context
      final searchResults = await _rag.search(
        text: query,
        limit: 3,
      );
      
      // Step 3: Build context from search results
      String context = '';
      if (searchResults.isNotEmpty) {
        context = 'Relevant information from documents:\n\n';
        for (var result in searchResults) {
          final docName = result.chunk.document.target?.fileName ?? 'Unknown';
          context += '[$docName]: ${result.chunk.content}\n\n';
        }
      }
      
      // Step 4: Build prompt with context
      final prompt = context.isEmpty 
          ? query 
          : '''$context

Based on the information above, please answer this question:
$query''';
      
      // Step 5: Get LLM response
      final response = await _chatModel.generateCompletion(
        messages: [
          ChatMessage(content: prompt, role: 'user'),
        ],
      );
      
      // Add AI response
      setState(() {
        _messages.add(AppMessage(
          text: response.response,
          isUser: false,
        ));
      });
    } catch (e) {
      setState(() {
        _messages.add(AppMessage(
          text: 'Error: $e',
          isUser: false,
        ));
      });
    }
  }

  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mobile RAG'),
        actions: [
          if (_isReady) ...[
            // Import status indicator
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
            // Library menu
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
                      const Icon(Icons.add, size: 20),
                      const SizedBox(width: 12),
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
                // Chat messages area
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
                            return MessageBubble(message: message);
                          },
                        ),
                ),
                DocumentPreview(
                  pendingDocs: _pendingDocs,
                  onRemove: _removePendingDoc,
                ),
                // Input area with add document button
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
