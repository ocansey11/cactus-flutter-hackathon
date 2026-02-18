import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/similarity_entity.dart';
import '../objectbox.g.dart';

class DocumentSimilarity {
  final String id;
  final String projectId;
  final String doc1Id;
  final String doc2Id;
  final double similarityScore;
  final DateTime computedAt;
  final bool needsUpdate;

  DocumentSimilarity({
    required this.id,
    required this.projectId,
    required this.doc1Id,
    required this.doc2Id,
    required this.similarityScore,
    required this.computedAt,
    this.needsUpdate = false,
  });

  static String createPairKey(String docId1, String docId2) {
    final sorted = [docId1, docId2]..sort();
    return '${sorted[0]}:${sorted[1]}';
  }

  String get pairKey => createPairKey(doc1Id, doc2Id);

  factory DocumentSimilarity.fromEntity(SimilarityEntity entity) {
    return DocumentSimilarity(
      id: entity.id,
      projectId: entity.projectId,
      doc1Id: entity.doc1Id,
      doc2Id: entity.doc2Id,
      similarityScore: entity.similarityScore,
      computedAt: DateTime.fromMillisecondsSinceEpoch(entity.computedAt),
      needsUpdate: entity.needsUpdate,
    );
  }
}

class SimilarityCache {
  static const _uuid = Uuid();

  void storeSimilarity({
    required String projectId,
    required String doc1Id,
    required String doc2Id,
    required double similarityScore,
  }) {
    final sorted = [doc1Id, doc2Id]..sort();

    final query = ObjectBoxManager.similarities
        .query(SimilarityEntity_.projectId.equals(projectId) &
            SimilarityEntity_.doc1Id.equals(sorted[0]) &
            SimilarityEntity_.doc2Id.equals(sorted[1]))
        .build();
    final existing = query.findFirst();
    query.close();

    if (existing != null) {
      existing.similarityScore = similarityScore;
      existing.computedAt = DateTime.now().millisecondsSinceEpoch;
      existing.needsUpdate = false;
      ObjectBoxManager.similarities.put(existing);
      return;
    }

    ObjectBoxManager.similarities.put(SimilarityEntity(
      id: _uuid.v4(),
      projectId: projectId,
      doc1Id: sorted[0],
      doc2Id: sorted[1],
      similarityScore: similarityScore,
      computedAt: DateTime.now().millisecondsSinceEpoch,
      needsUpdate: false,
    ));
  }

  List<DocumentSimilarity> getSimilaritiesForProject(String projectId,
      {double threshold = 0.0}) {
    final query = ObjectBoxManager.similarities
        .query(SimilarityEntity_.projectId.equals(projectId) &
            SimilarityEntity_.similarityScore.greaterOrEqual(threshold))
        .build();
    final entities = query.find();
    query.close();
    return entities.map((e) => DocumentSimilarity.fromEntity(e)).toList();
  }

  DocumentSimilarity? getSimilarity(
      String projectId, String doc1Id, String doc2Id) {
    final sorted = [doc1Id, doc2Id]..sort();
    final query = ObjectBoxManager.similarities
        .query(SimilarityEntity_.projectId.equals(projectId) &
            SimilarityEntity_.doc1Id.equals(sorted[0]) &
            SimilarityEntity_.doc2Id.equals(sorted[1]))
        .build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? DocumentSimilarity.fromEntity(entity) : null;
  }

  void markDocumentDirty(String projectId, String docId) {
    final q1 = ObjectBoxManager.similarities
        .query(SimilarityEntity_.projectId.equals(projectId) &
            SimilarityEntity_.doc1Id.equals(docId))
        .build();
    final q2 = ObjectBoxManager.similarities
        .query(SimilarityEntity_.projectId.equals(projectId) &
            SimilarityEntity_.doc2Id.equals(docId))
        .build();

    for (final entity in [...q1.find(), ...q2.find()]) {
      entity.needsUpdate = true;
      ObjectBoxManager.similarities.put(entity);
    }
    q1.close();
    q2.close();
  }

  void deleteForProject(String projectId) {
    final query = ObjectBoxManager.similarities
        .query(SimilarityEntity_.projectId.equals(projectId))
        .build();
    ObjectBoxManager.similarities.removeMany(query.findIds());
    query.close();
  }
}
