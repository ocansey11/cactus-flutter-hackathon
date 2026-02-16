import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import 'package:cactus/memory/similarity_cache.dart';  
import 'tool_registry.dart';
import 'dart:convert';

/// Document Similarity Tool - analyzes relationships between documents
class DocumentSimilarityTool implements ToolHandler {
  @override
  CactusTool get definition => createTool(
    'compute_document_similarity',
    'Analyze similarity relationships between documents in the knowledge base',
    {
      'threshold': ToolParameter(
        type: 'number',
        description: 'Minimum similarity score to include (0.0-1.0, default: 0.5)',
        required: false,
      ),
      'force_recompute': ToolParameter(
        type: 'boolean',
        description: 'Force recomputation even if cache exists (default: false)',
        required: false,
      ),
    },
  );

  @override
  Future<String> call(Map<String, dynamic> args, RAGService? rag) async {
    if (rag == null) {
      return 'Error: RAG service not available';
    }
    
    final threshold = (args['threshold'] as num?)?.toDouble() ?? 0.5;
    final forceRecompute = args['force_recompute'] as bool? ?? false;
    
    // Get similarity cache from RAG service
    final cache = rag.similarityCache;
    
    // Get all documents with metadata
    final docsWithMetadata = await rag.getAllDocumentsWithMetadata();
    
    if (docsWithMetadata.isEmpty) {
      return 'No documents in knowledge base yet.';
    }
    
    if (docsWithMetadata.length == 1) {
      return 'Only one document in knowledge base. Need at least 2 documents to compute similarities.';
    }
    
    // Check if we can use cache
    if (!forceRecompute) {
      final cachedSimilarities = cache.getSimilaritiesAboveThreshold(threshold);
      
      // If cache is fresh and has results, use it
      if (cachedSimilarities.isNotEmpty) {
        final allDocsFresh = docsWithMetadata.every((docData) {
          final docId = docData['id'] as String;
          return cache.isCacheFresh(docId);
        });
        
        if (allDocsFresh) {
          return _formatCachedResults(cachedSimilarities, docsWithMetadata, threshold);
        }
      }
    }
    
    // Compute fresh similarities
    final similarities = <DocumentSimilarity>[];
    
    for (int i = 0; i < docsWithMetadata.length; i++) {
      for (int j = i + 1; j < docsWithMetadata.length; j++) {
        final doc1 = docsWithMetadata[i];
        final doc2 = docsWithMetadata[j];
        final doc1Id = doc1['id'] as String;
        final doc2Id = doc2['id'] as String;
        
        // Search using doc2's content
        final doc2Content = (doc2['document'] as Document).content;
        final queryText = doc2Content.substring(
          0, 
          doc2Content.length > 500 ? 500 : doc2Content.length,
        );
        
        final results = await rag.search(query: queryText, limit: 5);
        
        if (results.isNotEmpty) {
          final avgDistance = results.map((r) => r.distance).reduce((a, b) => a + b) / results.length;
          final similarity = 1.0 - avgDistance;
          
          // Store in cache
          cache.storeSimilarity(
            doc1Id: doc1Id,
            doc2Id: doc2Id,
            similarityScore: similarity,
          );
          
          if (similarity >= threshold) {
            final docSim = DocumentSimilarity(
              id: '',
              doc1Id: doc1Id,
              doc2Id: doc2Id,
              similarityScore: similarity,
              computedAt: DateTime.now(),
            );
            similarities.add(docSim);
          }
        }
      }
    }
    
    // Mark all documents as clean
    for (final docData in docsWithMetadata) {
      final docId = docData['id'] as String;
      cache.markDocumentClean(docId);
    }
    
    if (similarities.isEmpty) {
      return 'No significant similarities found between documents (threshold: $threshold).';
    }
    
    return _formatResults(similarities, docsWithMetadata, threshold);
  }
  
  String _formatResults(
    List<DocumentSimilarity> similarities,
    List<Map<String, dynamic>> docsWithMetadata,
    double threshold,
  ) {
    // Create docId -> fileName map
    final docIdToName = <String, String>{};
    for (final docData in docsWithMetadata) {
      final docId = docData['id'] as String;
      final fileName = (docData['document'] as Document).fileName;
      docIdToName[docId] = fileName;
    }
    
    // Sort by similarity score (highest first)
    similarities.sort((a, b) => b.similarityScore.compareTo(a.similarityScore));
    
    final buffer = StringBuffer();
    buffer.writeln('Document Similarity Analysis:');
    buffer.writeln('Total documents: ${docsWithMetadata.length}');
    buffer.writeln('Similarity threshold: $threshold');
    buffer.writeln('\nRelationships found (${similarities.length}):');
    
    for (final sim in similarities) {
      final doc1Name = docIdToName[sim.doc1Id] ?? 'Unknown';
      final doc2Name = docIdToName[sim.doc2Id] ?? 'Unknown';
      final score = (sim.similarityScore * 100).toStringAsFixed(1);
      buffer.writeln('- $doc1Name ↔ $doc2Name: $score% similar');
    }
    
    return buffer.toString();
  }
  
  String _formatCachedResults(
    List<DocumentSimilarity> cachedSimilarities,
    List<Map<String, dynamic>> docsWithMetadata,
    double threshold,
  ) {
    final result = _formatResults(cachedSimilarities, docsWithMetadata, threshold);
    return '$result\n\n[Results from cache]';
  }
}