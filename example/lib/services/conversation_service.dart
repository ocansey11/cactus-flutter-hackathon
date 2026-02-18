import 'package:cactus/cactus.dart';
import 'rag_service.dart';
import 'chat_service.dart';
import 'document_service.dart';
import 'message_router.dart';
import 'project_service.dart' show ProjectService;
import '../tools/tool_registry.dart';

class ConversationService {
  final RAGService? ragService;
  final ChatService chatService;
  final MessageRouter? messageRouter;
  final ProjectService? projectService;
  late final ToolRegistry _toolRegistry;

  static const _directReturnTools = {
    'compute_document_similarity',
    'create_project_note',
  };

  ConversationService({
    this.ragService,
    required this.chatService,
    this.messageRouter,
    this.projectService,
  }) {
    _toolRegistry = ToolRegistry(
      ragService: ragService,
      projectService: projectService,
      chatModel: chatService.chatModel,
    );
  }

  Future<String> handleQuery({
    required String query,
    List<Map<String, dynamic>>? newDocs,
    String? projectName,
    String? conversationId,
  }) async {
    final hasPendingDocs = newDocs != null && newDocs.isNotEmpty;

    if (messageRouter == null || ragService == null) {
      return await handleSimpleChat(query: query);
    }

    final routeResult = await messageRouter!.route(
      query: query,
      hasPendingDocs: hasPendingDocs,
    );

    String? toolResult;
    final toolName = routeResult.functionResult?.toolName;

    if (routeResult.functionResult?.needsTool == true && toolName != null) {
      try {
        final args = Map<String, dynamic>.from(
            routeResult.functionResult!.parameters);
        if (conversationId != null) args['conversation_id'] = conversationId;

        toolResult = await _toolRegistry.execute(toolName, args);

        if (_directReturnTools.contains(toolName) && toolResult != null) {
          return toolResult;
        }
      } catch (_) {
        toolResult = null;
      }
    }

    switch (routeResult.messageType) {
      case MessageType.rag:
        return await _handleRAG(
          query: query,
          newDocs: newDocs ?? [],
          toolResult: toolResult,
          projectName: projectName,
        );
      case MessageType.simpleChat:
        return await _handleSimpleChat(query: query, toolResult: toolResult);
      case MessageType.toolCalling:
        return await handleSimpleChat(query: query);
    }
  }

  Future<String> _handleRAG({
    required String query,
    required List<Map<String, dynamic>> newDocs,
    String? toolResult,
    String? projectName,
  }) async {
    for (final doc in newDocs) {
      await ragService!.storeDocument(
        fileName: doc['fileName'],
        filePath: doc['filePath'] ?? '',
        content: doc['content'],
        fileSize: doc['fileSize'],
        projectName: projectName,
      );
    }

    final results =
        await ragService!.search(query: query, projectName: projectName, limit: 5);

    if (results.isEmpty) return 'No relevant content found in uploaded documents.';

    final context = ragService!.buildContext(results);
    final systemPrompt = ragService!.getSystemPrompt();
    final userPrompt = toolResult != null
        ? 'Tool result:\n$toolResult\n\nContext:\n$context\n\nQuestion: $query'
        : ragService!.createRAGPrompt(query, context);

    return await chatService.ragChat(
        query: query, context: userPrompt, systemPrompt: systemPrompt);
  }

  Future<String> _handleSimpleChat({
    required String query,
    String? toolResult,
  }) async {
    if (toolResult != null) {
      return await chatService.ragChat(
        query: query,
        context: 'Tool result:\n$toolResult\n\nQuestion: $query',
        systemPrompt:
            'You are a helpful assistant. Use the tool result to answer naturally.',
      );
    }
    return await chatService.simpleChat(query);
  }

  Future<String> handleSimpleChat({required String query}) async {
    return await chatService.simpleChat(query);
  }

  Future<BulkImportResult> bulkImport({
    required List<FileInfo> files,
    required Set<String> existingFileNames,
    String? projectName,
  }) async {
    if (ragService == null) throw Exception('RAG service not initialized');

    int addedCount = 0;
    int skippedCount = 0;
    final errors = <String>[];

    for (final file in files) {
      if (existingFileNames.contains(file.fileName)) {
        skippedCount++;
        continue;
      }
      try {
        final content =
            await DocumentService.extractContent(file.filePath, file.extension);
        if (!DocumentService.isValidContent(content)) {
          skippedCount++;
          continue;
        }
        await ragService!.storeDocument(
          fileName: file.fileName,
          filePath: file.filePath,
          content: content,
          fileSize: file.fileSize,
          projectName: projectName,
        );
        addedCount++;
      } catch (e) {
        errors.add('${file.fileName}: $e');
        skippedCount++;
      }
    }

    return BulkImportResult(
        addedCount: addedCount, skippedCount: skippedCount, errors: errors);
  }
}

class FileInfo {
  final String fileName;
  final String filePath;
  final String extension;
  final int fileSize;

  FileInfo({
    required this.fileName,
    required this.filePath,
    required this.extension,
    required this.fileSize,
  });
}

class BulkImportResult {
  final int addedCount;
  final int skippedCount;
  final List<String> errors;

  BulkImportResult({
    required this.addedCount,
    required this.skippedCount,
    required this.errors,
  });
}
