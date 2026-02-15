import 'package:cactus/cactus.dart';
import 'rag_service.dart';
import 'chat_service.dart';
import 'document_service.dart';
import 'message_router.dart';
import '../tools/tool_registry.dart';

class ConversationService {
  final RAGService? ragService;
  final ChatService chatService;
  final MessageRouter? messageRouter;
  
  ConversationService({
    this.ragService,
    required this.chatService,
    this.messageRouter,
  });
  
  Future<String> handleQuery({
    required String query,
    List<Map<String, dynamic>>? newDocs,
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
    if (routeResult.functionResult?.needsTool == true && 
        routeResult.functionResult?.toolName != null) {
      try {
        print('🔧 Executing tool: ${routeResult.functionResult!.toolName}');
        print('🔧 Parameters: ${routeResult.functionResult!.parameters}');
        toolResult = await ToolRegistry.executeTool(
          routeResult.functionResult!.toolName!,
          routeResult.functionResult!.parameters,
          ragService: ragService,
        );
        print('🔧 Tool result length: ${toolResult?.length ?? 0} chars');
        
        // If we have a tool result, return it directly for document similarity
        if (toolResult != null && routeResult.functionResult!.toolName == 'compute_document_similarity') {
          return toolResult;
        }
      } catch (e) {
        print('🔧 Tool execution error: $e');
        toolResult = null;
      }
    } else {
      print('🔧 No tool needed. Function result: ${routeResult.functionResult}');
    }
    
    switch (routeResult.messageType) {
      case MessageType.rag:
        return await _handleRAGWithTool(
          query: query,
          newDocs: newDocs ?? [],
          toolResult: toolResult,
        );
      
      case MessageType.simpleChat:
        return await _handleSimpleChatWithTool(
          query: query,
          toolResult: toolResult,
        );
      
      case MessageType.toolCalling:
        return await handleSimpleChat(query: query);
    }
  }
  
  Future<String> _handleRAGWithTool({
    required String query,
    required List<Map<String, dynamic>> newDocs,
    String? toolResult,
  }) async {
    if (newDocs.isNotEmpty) {
      for (final doc in newDocs) {
        await ragService!.storeDocument(
          fileName: doc['fileName'],
          filePath: doc['filePath'] ?? '',
          content: doc['content'],
          fileSize: doc['fileSize'],
        );
      }
    }
    
    final results = await ragService!.search(query: query, limit: 5);
    
    if (results.isEmpty) {
      return 'No relevant content found in the uploaded documents.';
    }
    
    final context = ragService!.buildContext(results);
    final systemPrompt = ragService!.getSystemPrompt();
    
    String userPrompt;
    if (toolResult != null) {
      userPrompt = '''Tool analysis result:
$toolResult

Document context:
$context

Question: $query

Answer the question using both the tool analysis and document context.''';
    } else {
      userPrompt = ragService!.createRAGPrompt(query, context);
    }
    
    return await chatService.ragChat(
      query: query,
      context: userPrompt,
      systemPrompt: systemPrompt,
    );
  }
  
  Future<String> _handleSimpleChatWithTool({
    required String query,
    String? toolResult,
  }) async {
    if (toolResult != null) {
      return await chatService.ragChat(
        query: query,
        context: '''Tool analysis result:
$toolResult

Question: $query

Answer the question using the tool analysis result above.''',
        systemPrompt: 'You are a helpful assistant. Use the tool result to answer the question naturally and conversationally.',
      );
    }
    
    return await chatService.simpleChat(query);
  }
  
  Future<String> handleRAGQuery({
    required String query,
    required List<Map<String, dynamic>> newDocs,
  }) async {
    if (ragService == null) {
      throw Exception('RAG service not initialized');
    }
    
    if (newDocs.isNotEmpty) {
      for (final doc in newDocs) {
        await ragService!.storeDocument(
          fileName: doc['fileName'],
          filePath: doc['filePath'] ?? '',
          content: doc['content'],
          fileSize: doc['fileSize'],
        );
      }
    }
    
    final results = await ragService!.search(query: query, limit: 5);
    
    if (results.isEmpty) {
      return 'No relevant content found in the uploaded documents.';
    }
    
    final context = ragService!.buildContext(results);
    final systemPrompt = ragService!.getSystemPrompt();
    final userPrompt = ragService!.createRAGPrompt(query, context);
    
    return await chatService.ragChat(
      query: query,
      context: userPrompt,
      systemPrompt: systemPrompt,
    );
  }
  
  Future<String> handleSimpleChat({required String query}) async {
    return await chatService.simpleChat(query);
  }
  
  Future<BulkImportResult> bulkImport({
    required List<FileInfo> files,
    required Set<String> existingFileNames,
  }) async {
    if (ragService == null) {
      throw Exception('RAG service not initialized');
    }
    
    int addedCount = 0;
    int skippedCount = 0;
    final errors = <String>[];
    
    for (final file in files) {
      if (existingFileNames.contains(file.fileName)) {
        skippedCount++;
        continue;
      }
      
      try {
        final content = await DocumentService.extractContent(
          file.filePath,
          file.extension,
        );
        
        if (!DocumentService.isValidContent(content)) {
          skippedCount++;
          continue;
        }
        
        await ragService!.storeDocument(
          fileName: file.fileName,
          filePath: file.filePath,
          content: content,
          fileSize: file.fileSize,
        );
        
        addedCount++;
      } catch (e) {
        errors.add('${file.fileName}: $e');
        skippedCount++;
      }
    }
    
    return BulkImportResult(
      addedCount: addedCount,
      skippedCount: skippedCount,
      errors: errors,
    );
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
