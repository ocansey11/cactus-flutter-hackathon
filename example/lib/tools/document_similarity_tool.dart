import 'dart:math';
import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';
import '../services/project_service.dart';
import '../models/document_graph.dart';
import 'tool_handler.dart';

class DocumentSimilarityTool implements ToolHandler {
  @override
  CactusTool get definition => CactusTool(
        name: 'compute_document_similarity',
        description:
            'Analyze similarity relationships between documents in the current project',
        parameters: ToolParametersSchema(
          properties: {
            'threshold': ToolParameter(
              type: 'number',
              description: 'Minimum similarity score to include (0.0-1.0)',
              required: false,
            ),
          },
        ),
      );

  @override
  Future<String> call(
    Map<String, dynamic> args, {
    RAGService? ragService,
    ProjectService? projectService,
    CactusLM? chatModel,
  }) async {
    if (ragService == null) return 'RAG service not available.';

    final threshold = (args['threshold'] as num?)?.toDouble() ?? 0.8;
    final projectName = projectService?.currentProject?.name;
    final documents = await ragService.getAllDocuments();

    final filtered = projectName != null
        ? documents
            .where((d) => d.chunks.isNotEmpty)
            .toList()
        : documents.where((d) => d.chunks.isNotEmpty).toList();

    if (filtered.length < 2) {
      return 'Need at least 2 documents with embeddings to compute similarities.';
    }

    final docEmbeddings = <String, List<double>>{};
    for (final doc in filtered) {
      final embedding = _averageEmbedding(doc);
      if (embedding.isNotEmpty) docEmbeddings[doc.fileName] = embedding;
    }

    final similarities = <Map<String, dynamic>>[];
    final docNames = docEmbeddings.keys.toList();

    for (int i = 0; i < docNames.length; i++) {
      for (int j = i + 1; j < docNames.length; j++) {
        final score = _cosineSimilarity(
          docEmbeddings[docNames[i]]!,
          docEmbeddings[docNames[j]]!,
        );
        if (score >= threshold) {
          similarities.add({
            'doc1': docNames[i],
            'doc2': docNames[j],
            'similarity': score,
          });
        }
      }
    }

    similarities.sort(
        (a, b) => (b['similarity'] as double).compareTo(a['similarity'] as double));

    DocumentGraphStore.lastGraph = DocumentGraph.fromSimilarities(
      similarities: similarities,
      allDocuments: docNames,
      threshold: threshold,
    );

    if (similarities.isEmpty) {
      return 'No similarities above ${(threshold * 100).toStringAsFixed(0)}% threshold across ${docNames.length} documents.';
    }

    final buffer = StringBuffer();
    buffer.writeln('Document Similarity Analysis');
    buffer.writeln('Documents: ${filtered.length}  Threshold: ${(threshold * 100).toStringAsFixed(0)}%');
    buffer.writeln('Relationships: ${similarities.length}');
    buffer.writeln();

    for (final sim in similarities) {
      final score = ((sim['similarity'] as double) * 100).toStringAsFixed(1);
      buffer.writeln('${sim['doc1']} <-> ${sim['doc2']}: $score%');
    }

    return buffer.toString();
  }

  List<double> _averageEmbedding(Document doc) {
    final chunks = doc.chunks.toList();
    if (chunks.isEmpty) return [];
    final dims = chunks.first.embeddings.length;
    final avg = List<double>.filled(dims, 0.0);
    for (final chunk in chunks) {
      for (int i = 0; i < dims; i++) avg[i] += chunk.embeddings[i];
    }
    for (int i = 0; i < dims; i++) avg[i] /= chunks.length;
    return avg;
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0, magA = 0, magB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }
    if (magA == 0 || magB == 0) return 0.0;
    return dot / (sqrt(magA) * sqrt(magB));
  }
}

class DocumentGraphStore {
  static DocumentGraph? lastGraph;
}
