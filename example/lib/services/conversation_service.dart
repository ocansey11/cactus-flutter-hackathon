import 'package:cactus/cactus.dart';
import 'rag_service.dart';
import 'chat_service.dart';
import 'document_service.dart';

class ConversationService {
  final RAGService ragService;
  final ChatService chatService;
  
  ConversationService({
    required this.ragService,
    required this.chatService,
  });
  
  Future<String> handleRAGQuery({
    required String query,
    required List<Map<String, dynamic>> newDocs,
  }) async {
    if (newDocs.isNotEmpty) {
      for (final doc in newDocs) {
        await ragService.storeDocument(
          fileName: doc['fileName'],
          filePath: doc['filePath'] ?? '',
          content: doc['content'],
          fileSize: doc['fileSize'],
        );
      }
    }
    
    final results = await ragService.search(query: query, limit: 5);
    
    if (results.isEmpty) {
      return 'No relevant content found in the uploaded documents.';
    }
    
    final context = ragService.buildContext(results);
    final systemPrompt = ragService.getSystemPrompt();
    final userPrompt = ragService.createRAGPrompt(query, context);
    
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
        
        await ragService.storeDocument(
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
