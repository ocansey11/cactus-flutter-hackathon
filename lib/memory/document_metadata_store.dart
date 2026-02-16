import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/document_metadata_entity.dart';
import 'package:cactus/objectbox.g.dart';

class DocumentMetadata {
  final String id;
  final String fileName;
  final String filePath;
  final int fileSize;
  final DateTime uploadedAt;
  final DateTime lastModified;
  final Map<String, dynamic> customMetadata;
  
  DocumentMetadata({
    required this.id,
    required this.fileName,
    required this.filePath,
    required this.fileSize,
    required this.uploadedAt,
    required this.lastModified,
    this.customMetadata = const {},
  });
  
  DocumentMetadata copyWith({
    String? id,
    String? fileName,
    String? filePath,
    int? fileSize,
    DateTime? uploadedAt,
    DateTime? lastModified,
    Map<String, dynamic>? customMetadata,
  }) {
    return DocumentMetadata(
      id: id ?? this.id,
      fileName: fileName ?? this.fileName,
      filePath: filePath ?? this.filePath,
      fileSize: fileSize ?? this.fileSize,
      uploadedAt: uploadedAt ?? this.uploadedAt,
      lastModified: lastModified ?? this.lastModified,
      customMetadata: customMetadata ?? this.customMetadata,
    );
  }
  
  factory DocumentMetadata.fromEntity(DocumentMetadataEntity entity) {
    return DocumentMetadata(
      id: entity.id,
      fileName: entity.fileName,
      filePath: entity.filePath,
      fileSize: entity.fileSize,
      uploadedAt: DateTime.fromMillisecondsSinceEpoch(entity.uploadedAt),
      lastModified: DateTime.fromMillisecondsSinceEpoch(entity.lastModified),
      customMetadata: {},
    );
  }
}

class DocumentMetadataStore {
  static const _uuid = Uuid();
  
  String registerDocument({
    required String fileName,
    required String filePath,
    required int fileSize,
  }) {
    final query = ObjectBoxManager.metadata.query(
      DocumentMetadataEntity_.fileName.equals(fileName)
    ).build();
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
      fileName: fileName,
      filePath: filePath,
      fileSize: fileSize,
      uploadedAt: now.millisecondsSinceEpoch,
      lastModified: now.millisecondsSinceEpoch,
    );
    
    ObjectBoxManager.metadata.put(entity);
    return entity.id;
  }
  
  String? getDocIdByFileName(String fileName) {
    final query = ObjectBoxManager.metadata.query(
      DocumentMetadataEntity_.fileName.equals(fileName)
    ).build();
    final entity = query.findFirst();
    query.close();
    return entity?.id;
  }
  
  DocumentMetadata? getMetadata(String docId) {
    final query = ObjectBoxManager.metadata.query(
      DocumentMetadataEntity_.id.equals(docId)
    ).build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? DocumentMetadata.fromEntity(entity) : null;
  }
  
  List<DocumentMetadata> getAllMetadata() {
    final entities = ObjectBoxManager.metadata.getAll();
    return entities.map((e) => DocumentMetadata.fromEntity(e)).toList();
  }
  
  void renameDocument(String docId, String newFileName) {
    final query = ObjectBoxManager.metadata.query(
      DocumentMetadataEntity_.id.equals(docId)
    ).build();
    final entity = query.findFirst();
    query.close();
    
    if (entity != null) {
      entity.fileName = newFileName;
      entity.lastModified = DateTime.now().millisecondsSinceEpoch;
      ObjectBoxManager.metadata.put(entity);
    }
  }
  
  void deleteDocument(String docId) {
    final query = ObjectBoxManager.metadata.query(
      DocumentMetadataEntity_.id.equals(docId)
    ).build();
    final entity = query.findFirst();
    query.close();
    
    if (entity != null) {
      ObjectBoxManager.metadata.remove(entity.objectId);
    }
  }
  
  void updateCustomMetadata(String docId, Map<String, dynamic> customData) {
    final query = ObjectBoxManager.metadata.query(
      DocumentMetadataEntity_.id.equals(docId)
    ).build();
    final entity = query.findFirst();
    query.close();
    
    if (entity != null) {
      entity.customMetadata = customData.toString();
      entity.lastModified = DateTime.now().millisecondsSinceEpoch;
      ObjectBoxManager.metadata.put(entity);
    }
  }
  
  Future<void> loadFromDisk() async {}
  
  void dispose() {}
}