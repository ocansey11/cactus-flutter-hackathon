import 'package:objectbox/objectbox.dart';

@Entity()
class DocumentMetadataEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;

  // belongs to a project
  @Index()
  String projectId;

  String fileName;
  String filePath;
  int fileSize;
  int uploadedAt;
  int lastModified;

  // optional extra metadata (e.g. abstract, authors, tags)
  String? customMetadata;

  DocumentMetadataEntity({
    required this.id,
    required this.projectId,
    required this.fileName,
    required this.filePath,
    required this.fileSize,
    required this.uploadedAt,
    required this.lastModified,
    this.customMetadata,
  });
}
