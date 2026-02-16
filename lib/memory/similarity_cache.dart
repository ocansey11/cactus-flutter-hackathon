import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/similarity_entity.dart';
import 'package:cactus/objectbox.g.dart';

class DocumentSimilarity {
  final String id;
  final String doc1Id;
  final String doc2Id;
  final double similarityScore;
  final DateTime computedAt;
  final bool needsUpdate;
  
  DocumentSimilarity({
    required this.id,
    required this.doc1Id,
    required this.doc2Id,
    required this.similarityScore,
    required this.computedAt,
    this.needsUpdate = false,
  });
  
  DocumentSimilarity copyWith({
    String? id,
    String? doc1Id,
    String? doc2Id,
    double? similarityScore,
    DateTime? computedAt,
    bool? needsUpdate,
  }) {
    return DocumentSimilarity(
      id: id ?? this.id,
      doc1Id: doc1Id ?? this.doc1Id,
      doc2Id: doc2Id ?? this.doc2Id,
      similarityScore: similarityScore ?? this.similarityScore,
      computedAt: computedAt ?? this.computedAt,
      needsUpdate: needsUpdate ?? this.needsUpdate,
    );
  }
  
  static String createPairKey(String docId1, String docId2) {
    final sorted = [docId1, docId2]..sort();
    return '${sorted[0]}:${sorted[1]}';
  }
  
  String get pairKey => createPairKey(doc1Id, doc2Id);
  
  factory DocumentSimilarity.fromEntity(SimilarityEntity entity) {
    return DocumentSimilarity(
      id: entity.id,
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
    required String doc1Id,
    required String doc2Id,
    required double similarityScore,
  }) {
    final pairKey = DocumentSimilarity.createPairKey(doc1Id, doc2Id);
    final sorted = pairKey.split(':');
    
    final query = ObjectBoxManager.similarities.query(
      SimilarityEntity_.doc1Id.equals(sorted[0]) & 
      SimilarityEntity_.doc2Id.equals(sorted[1])
    ).build();
    final existing = query.findFirst();
    query.close();
    
    if (existing != null) {
      existing.similarityScore = similarityScore;
      existing.computedAt = DateTime.now().millisecondsSinceEpoch;
      existing.needsUpdate = false;
      ObjectBoxManager.similarities.put(existing);
      return;
    }
    
    final entity = SimilarityEntity(
      id: _uuid.v4(),
      doc1Id: sorted[0],
      doc2Id: sorted[1],
      similarityScore: similarityScore,
      computedAt: DateTime.now().millisecondsSinceEpoch,
      needsUpdate: false,
    );
    
    ObjectBoxManager.similarities.put(entity);
  }
  
  DocumentSimilarity? getSimilarity(String doc1Id, String doc2Id) {
    final pairKey = DocumentSimilarity.createPairKey(doc1Id, doc2Id);
    final sorted = pairKey.split(':');
    
    final query = ObjectBoxManager.similarities.query(
      SimilarityEntity_.doc1Id.equals(sorted[0]) & 
      SimilarityEntity_.doc2Id.equals(sorted[1])
    ).build();
    final entity = query.findFirst();
    query.close();
    
    return entity != null ? DocumentSimilarity.fromEntity(entity) : null;
  }
  
  List<DocumentSimilarity> getSimilaritiesForDocument(String docId) {
    final query1 = ObjectBoxManager.similarities.query(
      SimilarityEntity_.doc1Id.equals(docId)
    ).build();
    final entities1 = query1.find();
    query1.close();
    
    final query2 = ObjectBoxManager.similarities.query(
      SimilarityEntity_.doc2Id.equals(docId)
    ).build();
    final entities2 = query2.find();
    query2.close();
    
    final all = [...entities1, ...entities2];
    return all.map((e) => DocumentSimilarity.fromEntity(e)).toList();
  }
  
  List<DocumentSimilarity> getSimilaritiesAboveThreshold(double threshold) {
    final query = ObjectBoxManager.similarities.query(
      SimilarityEntity_.similarityScore.greaterOrEqual(threshold)
    ).build();
    final entities = query.find();
    query.close();
    return entities.map((e) => DocumentSimilarity.fromEntity(e)).toList();
  }
  
  List<DocumentSimilarity> getAllSimilarities() {
    final entities = ObjectBoxManager.similarities.getAll();
    return entities.map((e) => DocumentSimilarity.fromEntity(e)).toList();
  }
  
  void markDocumentDirty(String docId) {
    final sims = getSimilaritiesForDocument(docId);
    
    for (final sim in sims) {
      final query = ObjectBoxManager.similarities.query(
        SimilarityEntity_.id.equals(sim.id)
      ).build();
      final entity = query.findFirst();
      query.close();
      
      if (entity != null) {
        entity.needsUpdate = true;
        ObjectBoxManager.similarities.put(entity);
      }
    }
  }
  
  Set<String> getDirtyDocuments() {
    final query = ObjectBoxManager.similarities.query(
      SimilarityEntity_.needsUpdate.equals(true)
    ).build();
    final entities = query.find();
    query.close();
    
    final dirtyDocs = <String>{};
    for (final entity in entities) {
      dirtyDocs.add(entity.doc1Id);
      dirtyDocs.add(entity.doc2Id);
    }
    return dirtyDocs;
  }
  
  void markDocumentClean(String docId) {
    final sims = getSimilaritiesForDocument(docId);
    
    for (final sim in sims) {
      final query = ObjectBoxManager.similarities.query(
        SimilarityEntity_.id.equals(sim.id)
      ).build();
      final entity = query.findFirst();
      query.close();
      
      if (entity != null) {
        entity.needsUpdate = false;
        ObjectBoxManager.similarities.put(entity);
      }
    }
  }
  
  void deleteSimilaritiesForDocument(String docId) {
    final query1 = ObjectBoxManager.similarities.query(
      SimilarityEntity_.doc1Id.equals(docId)
    ).build();
    ObjectBoxManager.similarities.removeMany(query1.findIds());
    query1.close();
    
    final query2 = ObjectBoxManager.similarities.query(
      SimilarityEntity_.doc2Id.equals(docId)
    ).build();
    ObjectBoxManager.similarities.removeMany(query2.findIds());
    query2.close();
  }
  
  bool isCacheFresh(String docId) {
    final sims = getSimilaritiesForDocument(docId);
    return !sims.any((s) => s.needsUpdate);
  }
  
  void clearCache() {
    ObjectBoxManager.similarities.removeAll();
  }
  
  Map<String, dynamic> getStats() {
    final all = getAllSimilarities();
    final needUpdate = all.where((s) => s.needsUpdate).length;
    
    return {
      'totalSimilarities': all.length,
      'needingUpdate': needUpdate,
      'dirtyDocuments': getDirtyDocuments().length,
      'cacheHitRate': needUpdate == 0 ? 1.0 : 1.0 - (needUpdate / all.length),
    };
  }
  
  Future<void> loadFromDisk() async {}
  
  void dispose() {}
}