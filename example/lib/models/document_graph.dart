/// Represents a node in the document similarity graph
class DocumentNode {
  final String id;
  final String fileName;
  final int connectionCount;

  DocumentNode({
    required this.id,
    required this.fileName,
    this.connectionCount = 0,
  });
}

/// Represents an edge (connection) between two documents
class DocumentEdge {
  final String sourceId;
  final String targetId;
  final double similarity;

  DocumentEdge({
    required this.sourceId,
    required this.targetId,
    required this.similarity,
  });
}

/// Complete graph data structure
class DocumentGraph {
  final List<DocumentNode> nodes;
  final List<DocumentEdge> edges;
  final double threshold;

  DocumentGraph({
    required this.nodes,
    required this.edges,
    required this.threshold,
  });

  factory DocumentGraph.fromSimilarities({
    required List<Map<String, dynamic>> similarities,
    required List<String> allDocuments,
    required double threshold,
  }) {
    // Create nodes
    final nodeMap = <String, int>{};
    for (var doc in allDocuments) {
      nodeMap[doc] = 0;
    }

    // Count connections for each node
    for (var sim in similarities) {
      final doc1 = sim['doc1'] as String;
      final doc2 = sim['doc2'] as String;
      nodeMap[doc1] = (nodeMap[doc1] ?? 0) + 1;
      nodeMap[doc2] = (nodeMap[doc2] ?? 0) + 1;
    }

    // Create node list
    final nodes = nodeMap.entries.map((entry) {
      return DocumentNode(
        id: entry.key,
        fileName: entry.key,
        connectionCount: entry.value,
      );
    }).toList();

    // Create edge list
    final edges = similarities.map((sim) {
      return DocumentEdge(
        sourceId: sim['doc1'] as String,
        targetId: sim['doc2'] as String,
        similarity: sim['similarity'] as double,
      );
    }).toList();

    return DocumentGraph(
      nodes: nodes,
      edges: edges,
      threshold: threshold,
    );
  }
}
