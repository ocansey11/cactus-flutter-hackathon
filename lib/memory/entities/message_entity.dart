import 'package:objectbox/objectbox.dart';

@Entity()
class MessageEntity {
  @Id()
  int objectId = 0;
  
  @Unique()
  String id;
  String conversationId;
  String text;
  bool isUser;
  List<double>? embedding;
  int timestamp;
  String? metadata;
  
  @Index()
  String get conversationIdIndex => conversationId;
  
  MessageEntity({
    required this.id,
    required this.conversationId,
    required this.text,
    required this.isUser,
    this.embedding,
    required this.timestamp,
    this.metadata,
  });
}

