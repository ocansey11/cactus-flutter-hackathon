import 'package:cactus/cactus.dart';
import '../prompts/prompts.dart';

class RAGService {
  final CactusRAG _rag;
  final CactusLM _embeddingModel;
  
  RAGService({
    required CactusRAG rag,
    required CactusLM embeddingModel,
  })  : _rag = rag,
        _embeddingModel = embeddingModel;
  
  CactusRAG get rag => _rag;
  
  Future<void> initialize() async {
    await _rag.initialize();
    _rag.setEmbeddingGenerator((text) async {
      final result = await _embeddingModel.generateEmbedding(text: text);
      return result.embeddings;
    });
    _rag.setChunking(chunkSize: 1024, chunkOverlap: 128);
  }
  
  Future<void> storeDocument({
    required String fileName,
    required String filePath,
    required String content,
    required int fileSize,
    String? projectName, // reserved for future project-scoped filtering
  }) async {
    await _rag.storeDocument(
      fileName: fileName,
      filePath: filePath,
      content: content,
      fileSize: fileSize,
    );
  }
  
  Future<List<ChunkSearchResult>> search({
    required String query,
    String? projectName, // reserved for future project-scoped filtering
    int limit = 5,
  }) async {
    return await _rag.search(
      text: query,
      limit: limit,
    );
  }
  
  Future<List<Document>> getAllDocuments() async {
    return await _rag.getAllDocuments();
  }
  
  String buildContext(List<ChunkSearchResult> results, {int maxLength = 2000}) {
    final contextChunks = results.map((r) => r.chunk.content).toList();
    var context = contextChunks.join('\n\n---\n\n');
    
    if (context.length > maxLength) {
      context = context.substring(0, maxLength) + '\n...[truncated for length]';
    }
    
    return context;
  }
  
  String createRAGPrompt(String query, String context) =>
      RAGPrompts.user(query: query, context: context);

  String createRAGPromptWithTool({
    required String query,
    required String context,
    required String toolResult,
  }) =>
      RAGPrompts.userWithToolResult(
          query: query, context: context, toolResult: toolResult);

  String getSystemPrompt() => RAGPrompts.system();
  
  Future<void> clearDatabase() async {
    await _rag.clearDatabase();
  }
  
  void close() {
    _rag.close();
  }
}
