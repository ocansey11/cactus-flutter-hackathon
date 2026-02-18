import 'package:objectbox/objectbox.dart';

@Entity()
class SimilarityEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;

  // both documents belong to the same project
  @Index()
  String projectId;

  String doc1Id;
  String doc2Id;
  double similarityScore;
  int computedAt;

  // flag to recompute when new papers are added
  bool needsUpdate;

  SimilarityEntity({
    required this.id,
    required this.projectId,
    required this.doc1Id,
    required this.doc2Id,
    required this.similarityScore,
    required this.computedAt,
    required this.needsUpdate,
  });
}
