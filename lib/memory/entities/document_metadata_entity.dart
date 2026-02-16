// lib/memory/entities/document_metadata_entity.dart
import 'package:objectbox/objectbox.dart';
@Entity()
class DocumentMetadataEntity {
  @Id()
  int objectId = 0;
  
  @Unique()
  String id;
  String fileName;
  String filePath;
  int fileSize;
  int uploadedAt;
  int lastModified;
  String? customMetadata;
  
  DocumentMetadataEntity({
    required this.id,
    required this.fileName,
    required this.filePath,
    required this.fileSize,
    required this.uploadedAt,
    required this.lastModified,
    this.customMetadata,
  });
}

