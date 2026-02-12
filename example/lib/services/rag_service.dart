import 'package:cactus/cactus.dart';

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
    int limit = 5,
  }) async {
    return await _rag.search(text: query, limit: limit);
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
  
  String createRAGPrompt(String query, String context) {
    return '''Here is the content from the uploaded document:

CONTEXT START:
$context
CONTEXT END:

Now answer this question using ONLY the information above: $query

Remember: Only use information from the CONTEXT section above. Do not add information from your training data.''';
  }
  
  String getSystemPrompt() {
    return 'You are a document Q&A assistant. You must ONLY use the information provided in the Context section below. DO NOT use your general knowledge. If the answer is not in the Context, say "I cannot find that information in the provided document."';
  }
  
  void close() {
    _rag.close();
  }
}
