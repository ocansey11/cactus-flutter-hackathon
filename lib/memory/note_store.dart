import 'dart:convert';
import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/note_entity.dart';
import '../objectbox.g.dart';

class Note {
  final String id;
  final String projectId;
  final String conversationId;
  final String title;
  final String content;
  final String noteType;
  final List<String> referencedPapers;
  final DateTime createdAt;
  final DateTime updatedAt;

  Note({
    required this.id,
    required this.projectId,
    required this.conversationId,
    required this.title,
    required this.content,
    required this.noteType,
    required this.referencedPapers,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Note.fromEntity(NoteEntity entity) {
    List<String> papers = [];
    try {
      papers = List<String>.from(jsonDecode(entity.referencedPapers));
    } catch (_) {}

    return Note(
      id: entity.id,
      projectId: entity.projectId,
      conversationId: entity.conversationId,
      title: entity.title,
      content: entity.content,
      noteType: entity.noteType,
      referencedPapers: papers,
      createdAt: DateTime.fromMillisecondsSinceEpoch(entity.createdAt),
      updatedAt: DateTime.fromMillisecondsSinceEpoch(entity.updatedAt),
    );
  }
}

class NoteStore {
  static const _uuid = Uuid();

  Note createNote({
    required String projectId,
    required String conversationId,
    required String title,
    required String content,
    required String noteType,
    List<String> referencedPapers = const [],
  }) {
    final now = DateTime.now();
    final entity = NoteEntity(
      id: _uuid.v4(),
      projectId: projectId,
      conversationId: conversationId,
      title: title,
      content: content,
      noteType: noteType,
      referencedPapers: jsonEncode(referencedPapers),
      createdAt: now.millisecondsSinceEpoch,
      updatedAt: now.millisecondsSinceEpoch,
    );
    ObjectBoxManager.notes.put(entity);
    return Note.fromEntity(entity);
  }

  List<Note> getNotesForProject(String projectId) {
    final query = ObjectBoxManager.notes
        .query(NoteEntity_.projectId.equals(projectId))
        .build();
    final entities = query.find();
    query.close();
    entities.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return entities.map((e) => Note.fromEntity(e)).toList();
  }

  Note? getNoteForConversation(String conversationId) {
    final query = ObjectBoxManager.notes
        .query(NoteEntity_.conversationId.equals(conversationId))
        .build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? Note.fromEntity(entity) : null;
  }

  void updateNote({
    required String noteId,
    String? title,
    String? content,
    String? noteType,
    List<String>? referencedPapers,
  }) {
    final query = ObjectBoxManager.notes
        .query(NoteEntity_.id.equals(noteId))
        .build();
    final entity = query.findFirst();
    query.close();

    if (entity == null) return;
    if (title != null) entity.title = title;
    if (content != null) entity.content = content;
    if (noteType != null) entity.noteType = noteType;
    if (referencedPapers != null) {
      entity.referencedPapers = jsonEncode(referencedPapers);
    }
    entity.updatedAt = DateTime.now().millisecondsSinceEpoch;
    ObjectBoxManager.notes.put(entity);
  }

  void deleteNote(String noteId) {
    final query = ObjectBoxManager.notes
        .query(NoteEntity_.id.equals(noteId))
        .build();
    final entity = query.findFirst();
    query.close();
    if (entity != null) ObjectBoxManager.notes.remove(entity.objectId);
  }

  void deleteAllForProject(String projectId) {
    final query = ObjectBoxManager.notes
        .query(NoteEntity_.projectId.equals(projectId))
        .build();
    ObjectBoxManager.notes.removeMany(query.findIds());
    query.close();
  }
}
