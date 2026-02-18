import 'package:objectbox/objectbox.dart';

@Entity()
class NoteEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;

  // belongs to a project
  @Index()
  String projectId;

  // linked to the conversation that generated this note
  @Index()
  String conversationId;

  String title;
  String content;

  // concept | summary | writeup | auto
  String noteType;

  // JSON-encoded list of paper file names used as RAG sources
  // e.g. '["paper1.pdf", "paper2.pdf"]'
  // read-only — only updated by the model via tools
  String referencedPapers;

  int createdAt;
  int updatedAt;

  NoteEntity({
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
}
