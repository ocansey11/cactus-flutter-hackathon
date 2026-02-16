import 'package:objectbox/objectbox.dart';

@Entity()
class SimilarityEntity {
  @Id()
  int objectId = 0;
  
  @Unique()
  String id;
  String doc1Id;
  String doc2Id;
  double similarityScore;
  int computedAt;
  bool needsUpdate;
  
  SimilarityEntity({
    required this.id,
    required this.doc1Id,
    required this.doc2Id,
    required this.similarityScore,
    required this.computedAt,
    required this.needsUpdate,
  });
}