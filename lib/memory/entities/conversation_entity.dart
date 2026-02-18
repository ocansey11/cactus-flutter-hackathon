import 'package:objectbox/objectbox.dart';

@Entity()
class ConversationEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;

  String title;

  // null = global conversation, set = belongs to a project
  @Index()
  String? projectId;

  bool isVoiceMode;

  int createdAt;
  int updatedAt;

  ConversationEntity({
    required this.id,
    required this.title,
    this.projectId,
    required this.isVoiceMode,
    required this.createdAt,
    required this.updatedAt,
  });
}
