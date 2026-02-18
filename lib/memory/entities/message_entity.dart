import 'package:objectbox/objectbox.dart';

@Entity()
class MessageEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;

  @Index()
  String conversationId;

  String text;
  bool isUser;
  int timestamp;

  // optional embedding for semantic search over messages later
  List<double>? embedding;

  // optional metadata (e.g. which tool was called, RAG sources used)
  String? metadata;

  MessageEntity({
    required this.id,
    required this.conversationId,
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.embedding,
    this.metadata,
  });
}
