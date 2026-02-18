import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/document_metadata_entity.dart';
import '../objectbox.g.dart';

class DocumentMetadata {
  final String id;
  final String projectId;
  final String fileName;
  final String filePath;
  final int fileSize;
  final DateTime uploadedAt;
  final DateTime lastModified;
  final Map<String, dynamic> customMetadata;

  DocumentMetadata({
    required this.id,
    required this.projectId,
    required this.fileName,
    required this.filePath,
    required this.fileSize,
    required this.uploadedAt,
    required this.lastModified,
    this.customMetadata = const {},
  });

  factory DocumentMetadata.fromEntity(DocumentMetadataEntity entity) {
    return DocumentMetadata(
      id: entity.id,
      projectId: entity.projectId,
      fileName: entity.fileName,
      filePath: entity.filePath,
      fileSize: entity.fileSize,
      uploadedAt: DateTime.fromMillisecondsSinceEpoch(entity.uploadedAt),
      lastModified: DateTime.fromMillisecondsSinceEpoch(entity.lastModified),
    );
  }
}

class DocumentMetadataStore {
  static const _uuid = Uuid();

  String registerDocument({
    required String projectId,
    required String fileName,
    required String filePath,
    required int fileSize,
  }) {
    final query = ObjectBoxManager.metadata
        .query(DocumentMetadataEntity_.fileName.equals(fileName) &
            DocumentMetadataEntity_.projectId.equals(projectId))
        .build();
    final existing = query.findFirst();
    query.close();

    if (existing != null) {
      existing.filePath = filePath;
      existing.fileSize = fileSize;
      existing.lastModified = DateTime.now().millisecondsSinceEpoch;
      ObjectBoxManager.metadata.put(existing);
      return existing.id;
    }

    final now = DateTime.now();
    final entity = DocumentMetadataEntity(
      id: _uuid.v4(),
      projectId: projectId,
      fileName: fileName,
      filePath: filePath,
      fileSize: fileSize,
      uploadedAt: now.millisecondsSinceEpoch,
      lastModified: now.millisecondsSinceEpoch,
    );
    ObjectBoxManager.metadata.put(entity);
    return entity.id;
  }

  List<DocumentMetadata> getDocumentsForProject(String projectId) {
    final query = ObjectBoxManager.metadata
        .query(DocumentMetadataEntity_.projectId.equals(projectId))
        .build();
    final entities = query.find();
    query.close();
    return entities.map((e) => DocumentMetadata.fromEntity(e)).toList();
  }

  DocumentMetadata? getById(String docId) {
    final query = ObjectBoxManager.metadata
        .query(DocumentMetadataEntity_.id.equals(docId))
        .build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? DocumentMetadata.fromEntity(entity) : null;
  }

  void deleteDocument(String docId) {
    final query = ObjectBoxManager.metadata
        .query(DocumentMetadataEntity_.id.equals(docId))
        .build();
    final entity = query.findFirst();
    query.close();
    if (entity != null) ObjectBoxManager.metadata.remove(entity.objectId);
  }

  void deleteAllForProject(String projectId) {
    final query = ObjectBoxManager.metadata
        .query(DocumentMetadataEntity_.projectId.equals(projectId))
        .build();
    ObjectBoxManager.metadata.removeMany(query.findIds());
    query.close();
  }
}
