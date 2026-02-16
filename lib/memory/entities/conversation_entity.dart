// lib/memory/entities/conversation_entity.dart
import 'package:objectbox/objectbox.dart';

@Entity()
class ConversationEntity {
  @Id()
  int objectId = 0;
  
  @Unique()
  String id;
  String title;
  int createdAt;
  int updatedAt;
  bool isVoiceMode;
  
  ConversationEntity({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.updatedAt,
    required this.isVoiceMode,
  });
}

