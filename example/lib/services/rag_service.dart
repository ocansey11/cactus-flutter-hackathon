import 'package:cactus/cactus.dart';
import 'package:cactus/memory/document_metadata_store.dart';
import 'package:cactus/memory/similarity_cache.dart';
import 'package:cactus/memory/objectbox_manager.dart';


class RAGService {
  final CactusRAG _rag;
  final CactusLM _embeddingModel;
  final DocumentMetadataStore _metadataStore;
  final SimilarityCache _similarityCache;
  
  RAGService({
    required CactusRAG rag,
    required CactusLM embeddingModel,
    required DocumentMetadataStore metadataStore,
    required SimilarityCache similarityCache,
  })  : _rag = rag,
        _embeddingModel = embeddingModel,
        _metadataStore = metadataStore,
        _similarityCache = similarityCache;
  
  CactusRAG get rag => _rag;
  DocumentMetadataStore get metadataStore => _metadataStore;
  SimilarityCache get similarityCache => _similarityCache;
  
  Future<void> initialize() async {
    await _rag.initialize();
    _rag.setEmbeddingGenerator((text) async {
      final result = await _embeddingModel.generateEmbedding(text: text);
      return result.embeddings;
    });
    _rag.setChunking(chunkSize: 1024, chunkOverlap: 128);
    await _metadataStore.loadFromDisk();
    await _similarityCache.loadFromDisk();
  }
  
  /// Store document and mark cache as dirty
  Future<String> storeDocument({
    required String fileName,
    required String filePath,
    required String content,
    required int fileSize,
  }) async {
    final docId = _metadataStore.registerDocument(
      fileName: fileName,
      filePath: filePath,
      fileSize: fileSize,
    );
    
    await _rag.storeDocument(
      fileName: fileName,
      filePath: filePath,
      content: content,
      fileSize: fileSize,
    );
    
    // Mark this document as needing similarity recomputation
    _similarityCache.markDocumentDirty(docId);
    
    return docId;
  }
  
  /// Get document ID by fileName
  String? getDocumentId(String fileName) {
    return _metadataStore.getDocIdByFileName(fileName);
  }
  
  /// Get all documents with their stable IDs
  Future<List<Map<String, dynamic>>> getAllDocumentsWithMetadata() async {
    final cactusDocuments = await _rag.getAllDocuments();
    final result = <Map<String, dynamic>>[];
    
    for (final doc in cactusDocuments) {
      final docId = _metadataStore.getDocIdByFileName(doc.fileName);
      if (docId != null) {
        final metadata = _metadataStore.getMetadata(docId);
        result.add({
          'id': docId,
          'document': doc,
          'metadata': metadata,
        });
      }
    }
    
    return result;
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
    _metadataStore.dispose();
    _similarityCache.dispose();
  }
}
