import 'package:objectbox/objectbox.dart';

@Entity()
class NoteEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;
  
  @Index()
  String projectId;

  @Index()
  String conversationId;
  String title;
  String content;
  String noteType;
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
