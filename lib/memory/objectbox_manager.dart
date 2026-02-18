import 'package:objectbox/objectbox.dart';
import '../services/rag.dart';
import 'entities/project_entity.dart';
import 'entities/conversation_entity.dart';
import 'entities/message_entity.dart';
import 'entities/document_metadata_entity.dart';
import 'entities/note_entity.dart';
import 'entities/similarity_entity.dart';

class ObjectBoxManager {
  static Store? _store;

  static Box<ProjectEntity>? _projectBox;
  static Box<ConversationEntity>? _conversationBox;
  static Box<MessageEntity>? _messageBox;
  static Box<DocumentMetadataEntity>? _metadataBox;
  static Box<NoteEntity>? _noteBox;
  static Box<SimilarityEntity>? _similarityBox;

  static Future<void> initialize(CactusRAG rag) async {
    if (_store != null) return;
    _store = rag.store;
    _projectBox = _store!.box<ProjectEntity>();
    _conversationBox = _store!.box<ConversationEntity>();
    _messageBox = _store!.box<MessageEntity>();
    _metadataBox = _store!.box<DocumentMetadataEntity>();
    _noteBox = _store!.box<NoteEntity>();
    _similarityBox = _store!.box<SimilarityEntity>();
  }

  static Box<ProjectEntity> get projects => _projectBox!;
  static Box<ConversationEntity> get conversations => _conversationBox!;
  static Box<MessageEntity> get messages => _messageBox!;
  static Box<DocumentMetadataEntity> get metadata => _metadataBox!;
  static Box<NoteEntity> get notes => _noteBox!;
  static Box<SimilarityEntity> get similarities => _similarityBox!;

  static void close() {}
}
