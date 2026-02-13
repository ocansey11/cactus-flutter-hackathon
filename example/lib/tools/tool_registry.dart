import '../services/rag_service.dart';

class ToolRegistry {
  static Future<String> executeTool(
    String name,
    Map<String, dynamic> arguments, {
    RAGService? ragService,
  }) async {
    switch (name) {
      case 'compute_document_similarity':
        return await _computeDocumentSimilarity(arguments, ragService);
      default:
        throw Exception('Tool not found: $name');
    }
  }
  
  static Future<String> _computeDocumentSimilarity(
    Map<String, dynamic> arguments,
    RAGService? ragService,
  ) async {
    if (ragService == null) {
      return 'Error: RAG service not available';
    }
    
    final threshold = (arguments['threshold'] as num?)?.toDouble() ?? 0.5;
    
    final documents = await ragService.getAllDocuments();
    
    if (documents.isEmpty) {
      return 'No documents in knowledge base yet.';
    }
    
    if (documents.length == 1) {
      return 'Only one document in knowledge base. Need at least 2 documents to compute similarities.';
    }
    
    final similarities = <Map<String, dynamic>>[];
    
    for (int i = 0; i < documents.length; i++) {
      for (int j = i + 1; j < documents.length; j++) {
        final doc1 = documents[i];
        final doc2 = documents[j];
        
        final results = await ragService.search(query: doc2.fileName, limit: 5);
        
        if (results.isNotEmpty) {
          final avgDistance = results.map((r) => r.distance).reduce((a, b) => a + b) / results.length;
          final similarity = 1.0 - avgDistance;
          
          if (similarity >= threshold) {
            similarities.add({
              'doc1': doc1.fileName,
              'doc2': doc2.fileName,
              'similarity': similarity,
            });
          }
        }
      }
    }
    
    if (similarities.isEmpty) {
      return 'No significant similarities found between documents (threshold: $threshold).';
    }
    
    similarities.sort((a, b) => (b['similarity'] as double).compareTo(a['similarity'] as double));
    
    final buffer = StringBuffer();
    buffer.writeln('Document Similarity Analysis:');
    buffer.writeln('Total documents: ${documents.length}');
    buffer.writeln('Similarity threshold: $threshold');
    buffer.writeln('\nRelationships found (${similarities.length}):');
    
    for (final sim in similarities) {
      final score = ((sim['similarity'] as double) * 100).toStringAsFixed(1);
      buffer.writeln('- ${sim['doc1']} ↔ ${sim['doc2']}: $score% similar');
    }
    
    return buffer.toString();
  }
}
