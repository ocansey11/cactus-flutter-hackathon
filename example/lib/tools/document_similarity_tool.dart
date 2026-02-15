import 'dart:math';
import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../models/document_graph.dart';

class DocumentSimilarityTool {
  // Store the last computed graph for access by the UI
  static DocumentGraph? lastComputedGraph;
  
  static Map<String, dynamic> getDefinition() {
    return {
      'type': 'function',
      'function': {
        'name': 'compute_document_similarity',
        'description':
            'Analyze similarity relationships between documents in the knowledge base',
        'parameters': {
          'type': 'object',
          'properties': {
            'threshold': {
              'type': 'number',
              'description': 'Minimum similarity score to include (0.0-1.0)',
              'default': 0.8,
            },
          },
          'required': [],
        },
      },
    };
  }

  static Future<String> execute(
    Map<String, dynamic> arguments,
    RAGService? ragService,
  ) async {
    if (ragService == null) {
      return 'Error: RAG service not available';
    }

    final threshold = (arguments['threshold'] as num?)?.toDouble() ?? 0.8;

    final documents = await ragService.getAllDocuments();

    if (documents.isEmpty) {
      return 'No documents in knowledge base yet.';
    }

    if (documents.length == 1) {
      return 'Only one document in knowledge base. Need at least 2 documents to compute similarities.';
    }

    // Filter documents that have chunks with embeddings
    final docsWithEmbeddings =
        documents.where((doc) => doc.chunks.isNotEmpty).toList();

    if (docsWithEmbeddings.isEmpty) {
      return 'No documents with embeddings found.';
    }

    if (docsWithEmbeddings.length == 1) {
      return 'Only one document with embeddings found. Need at least 2 documents.';
    }

    final similarities = <Map<String, dynamic>>[];

    // Compute embeddings for each document
    final docEmbeddings = <String, List<double>>{};
    for (final doc in docsWithEmbeddings) {
      try {
        final embedding = _getDocumentEmbedding(doc);
        if (embedding.isNotEmpty) {
          docEmbeddings[doc.fileName] = embedding;
        }
      } catch (e) {
        // Skip documents with embedding errors
        continue;
      }
    }

    // Compare all pairs of documents
    final docNames = docEmbeddings.keys.toList();
    for (int i = 0; i < docNames.length; i++) {
      for (int j = i + 1; j < docNames.length; j++) {
        final doc1Name = docNames[i];
        final doc2Name = docNames[j];

        final embedding1 = docEmbeddings[doc1Name]!;
        final embedding2 = docEmbeddings[doc2Name]!;

        try {
          final similarity = cosineSimilarity(embedding1, embedding2);

          if (similarity >= threshold) {
            similarities.add({
              'doc1': doc1Name,
              'doc2': doc2Name,
              'similarity': similarity,
            });
          }
        } catch (e) {
          // Skip pairs with errors
          continue;
        }
      }
    }

    // Sort by similarity (highest first)
    similarities.sort((a, b) =>
        (b['similarity'] as double).compareTo(a['similarity'] as double));

    // Store graph data for visualization (even if no similarities above threshold)
    lastComputedGraph = DocumentGraph.fromSimilarities(
      similarities: similarities,
      allDocuments: docNames,
      threshold: threshold,
    );

    if (similarities.isEmpty) {
      return 'No similarities found above ${(threshold * 100).toStringAsFixed(0)}% threshold.\nAll ${docNames.length} documents shown as isolated nodes in graph.';
    }

    final buffer = StringBuffer();
    buffer.writeln('Document Similarity Analysis:');
    buffer.writeln('Total documents: ${docsWithEmbeddings.length}');
    buffer.writeln('Similarity threshold: ${(threshold * 100).toStringAsFixed(0)}%');
    buffer.writeln('\nRelationships found (${similarities.length}):');

    for (final sim in similarities) {
      final score = ((sim['similarity'] as double) * 100).toStringAsFixed(1);
      buffer.writeln('- ${sim['doc1']} ↔ ${sim['doc2']}: $score% similar');
    }
    
    buffer.writeln('\n📊 Tap "View Graph" to see visual representation');

    return buffer.toString();
  }

  /// Compute cosine similarity between two vectors
  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Vectors must have same dimensions');
    }

    double dotProduct = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magnitudeA += a[i] * a[i];
      magnitudeB += b[i] * b[i];
    }

    if (magnitudeA == 0.0 || magnitudeB == 0.0) {
      return 0.0;
    }

    magnitudeA = sqrt(magnitudeA);
    magnitudeB = sqrt(magnitudeB);

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /// Compute average embedding vector for a document from its chunks
  static List<double> _getDocumentEmbedding(Document doc) {
    if (doc.chunks.isEmpty) {
      return [];
    }

    final chunkList = doc.chunks.toList();
    final dimensions = chunkList.first.embeddings.length;
    final avgEmbedding = List<double>.filled(dimensions, 0.0);

    for (final chunk in chunkList) {
      for (int i = 0; i < dimensions; i++) {
        avgEmbedding[i] += chunk.embeddings[i];
      }
    }

    // Average the embeddings
    for (int i = 0; i < dimensions; i++) {
      avgEmbedding[i] /= chunkList.length;
    }

    return avgEmbedding;
  }
}
