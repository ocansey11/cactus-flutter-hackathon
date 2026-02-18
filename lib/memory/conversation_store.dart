import 'package:cactus/cactus.dart';
import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/conversation_entity.dart';
import 'entities/message_entity.dart';
import '../objectbox.g.dart';

class Message {
  final String id;
  final String conversationId;
  final String text;
  final bool isUser;
  final List<double>? embedding;
  final DateTime timestamp;
  final Map<String, dynamic>? metadata;

  Message({
    required this.id,
    required this.conversationId,
    required this.text,
    required this.isUser,
    this.embedding,
    required this.timestamp,
    this.metadata,
  });

  factory Message.fromEntity(MessageEntity entity) {
    return Message(
      id: entity.id,
      conversationId: entity.conversationId,
      text: entity.text,
      isUser: entity.isUser,
      embedding: entity.embedding,
      timestamp: DateTime.fromMillisecondsSinceEpoch(entity.timestamp),
    );
  }
}

class Conversation {
  final String id;
  final String title;
  final String? projectId;
  final bool isVoiceMode;
  final DateTime createdAt;
  final DateTime updatedAt;

  Conversation({
    required this.id,
    required this.title,
    this.projectId,
    required this.isVoiceMode,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Conversation.fromEntity(ConversationEntity entity) {
    return Conversation(
      id: entity.id,
      title: entity.title,
      projectId: entity.projectId,
      isVoiceMode: entity.isVoiceMode,
      createdAt: DateTime.fromMillisecondsSinceEpoch(entity.createdAt),
      updatedAt: DateTime.fromMillisecondsSinceEpoch(entity.updatedAt),
    );
  }
}

class ConversationStore {
  final CactusLM _embeddingModel;
  static const _uuid = Uuid();

  ConversationStore({required CactusLM embeddingModel})
      : _embeddingModel = embeddingModel;

  Future<Conversation> createConversation({
    String? title,
    String? projectId,
    bool isVoiceMode = false,
  }) async {
    final now = DateTime.now();
    final entity = ConversationEntity(
      id: _uuid.v4(),
      title: title ?? 'New Chat ${now.day}/${now.month}',
      projectId: projectId,
      isVoiceMode: isVoiceMode,
      createdAt: now.millisecondsSinceEpoch,
      updatedAt: now.millisecondsSinceEpoch,
    );
    ObjectBoxManager.conversations.put(entity);
    return Conversation.fromEntity(entity);
  }

  List<Conversation> getAllConversations({String? projectId}) {
    final entities = projectId != null
        ? ObjectBoxManager.conversations
            .query(ConversationEntity_.projectId.equals(projectId))
            .build()
            .find()
        : ObjectBoxManager.conversations.getAll();

    final convos = entities.map((e) => Conversation.fromEntity(e)).toList();
    convos.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return convos;
  }

  Conversation? getConversation(String conversationId) {
    final query = ObjectBoxManager.conversations
        .query(ConversationEntity_.id.equals(conversationId))
        .build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? Conversation.fromEntity(entity) : null;
  }

  void updateTitle(String conversationId, String newTitle) {
    final query = ObjectBoxManager.conversations
        .query(ConversationEntity_.id.equals(conversationId))
        .build();
    final entity = query.findFirst();
    query.close();

    if (entity == null) return;
    entity.title = newTitle;
    entity.updatedAt = DateTime.now().millisecondsSinceEpoch;
    ObjectBoxManager.conversations.put(entity);
  }

  void deleteConversation(String conversationId) {
    final query = ObjectBoxManager.conversations
        .query(ConversationEntity_.id.equals(conversationId))
        .build();
    final entity = query.findFirst();
    query.close();

    if (entity == null) return;

    final msgQuery = ObjectBoxManager.messages
        .query(MessageEntity_.conversationId.equals(conversationId))
        .build();
    ObjectBoxManager.messages.removeMany(msgQuery.findIds());
    msgQuery.close();

    ObjectBoxManager.conversations.remove(entity.objectId);
  }

  Future<Message> addMessage({
    required String conversationId,
    required String text,
    required bool isUser,
    Map<String, dynamic>? metadata,
  }) async {
    List<double>? embedding;
    try {
      final result = await _embeddingModel.generateEmbedding(text: text);
      embedding = result.embeddings;
    } catch (_) {}

    final entity = MessageEntity(
      id: _uuid.v4(),
      conversationId: conversationId,
      text: text,
      isUser: isUser,
      embedding: embedding,
      timestamp: DateTime.now().millisecondsSinceEpoch,
      metadata: metadata?.toString(),
    );
    ObjectBoxManager.messages.put(entity);

    final convoQuery = ObjectBoxManager.conversations
        .query(ConversationEntity_.id.equals(conversationId))
        .build();
    final convo = convoQuery.findFirst();
    convoQuery.close();

    if (convo != null) {
      convo.updatedAt = DateTime.now().millisecondsSinceEpoch;
      ObjectBoxManager.conversations.put(convo);
    }

    return Message.fromEntity(entity);
  }

  List<Message> getMessages(String conversationId) {
    final query = ObjectBoxManager.messages
        .query(MessageEntity_.conversationId.equals(conversationId))
        .build();
    final entities = query.find();
    query.close();
    entities.sort((a, b) => a.timestamp.compareTo(b.timestamp));
    return entities.map((e) => Message.fromEntity(e)).toList();
  }

  List<ChatMessage> toChatMessages(String conversationId) {
    return getMessages(conversationId).map((m) {
      return ChatMessage(content: m.text, role: m.isUser ? 'user' : 'assistant');
    }).toList();
  }

  String getContextString(String conversationId, {int lastN = 5}) {
    final messages = getMessages(conversationId);
    final recent = messages.length > lastN
        ? messages.sublist(messages.length - lastN)
        : messages;
    return recent
        .map((m) => '${m.isUser ? 'User' : 'Assistant'}: ${m.text}')
        .join('\n');
  }
}
