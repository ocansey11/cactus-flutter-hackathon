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
      metadata: entity.metadata != null ? {} : null,
    );
  }
}

class Conversation {
  final String id;
  final String title;
  final DateTime createdAt;
  final DateTime updatedAt;
  final bool isVoiceMode;
  
  Conversation({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.updatedAt,
    this.isVoiceMode = false,
  });
  
  Conversation copyWith({
    String? id,
    String? title,
    DateTime? createdAt,
    DateTime? updatedAt,
    bool? isVoiceMode,
  }) {
    return Conversation(
      id: id ?? this.id,
      title: title ?? this.title,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      isVoiceMode: isVoiceMode ?? this.isVoiceMode,
    );
  }
  
  factory Conversation.fromEntity(ConversationEntity entity) {
    return Conversation(
      id: entity.id,
      title: entity.title,
      createdAt: DateTime.fromMillisecondsSinceEpoch(entity.createdAt),
      updatedAt: DateTime.fromMillisecondsSinceEpoch(entity.updatedAt),
      isVoiceMode: entity.isVoiceMode,
    );
  }
}

class ConversationStore {
  final CactusLM _embeddingModel;
  static const _uuid = Uuid();
  
  ConversationStore({
    required CactusLM embeddingModel,
  }) : _embeddingModel = embeddingModel;
  
  Future<Conversation> createConversation({
    String? title,
    bool isVoiceMode = false,
  }) async {
    final now = DateTime.now();
    final entity = ConversationEntity(
      id: _uuid.v4(),
      title: title ?? 'New Chat ${now.day}/${now.month}',
      createdAt: now.millisecondsSinceEpoch,
      updatedAt: now.millisecondsSinceEpoch,
      isVoiceMode: isVoiceMode,
    );
    
    ObjectBoxManager.conversations.put(entity);
    return Conversation.fromEntity(entity);
  }
  
  List<Conversation> getAllConversations() {
    final entities = ObjectBoxManager.conversations.getAll();
    final convos = entities.map((e) => Conversation.fromEntity(e)).toList();
    convos.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return convos;
  }
  
  Conversation? getConversation(String conversationId) {
    final query = ObjectBoxManager.conversations.query(
      ConversationEntity_.id.equals(conversationId)
    ).build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? Conversation.fromEntity(entity) : null;
  }
  
  Future<void> updateConversationTitle(String conversationId, String newTitle) async {
    final query = ObjectBoxManager.conversations.query(
      ConversationEntity_.id.equals(conversationId)
    ).build();
    final entity = query.findFirst();
    query.close();
    
    if (entity != null) {
      entity.title = newTitle;
      entity.updatedAt = DateTime.now().millisecondsSinceEpoch;
      ObjectBoxManager.conversations.put(entity);
    }
  }
  
  Future<void> deleteConversation(String conversationId) async {
    final convoQuery = ObjectBoxManager.conversations.query(
      ConversationEntity_.id.equals(conversationId)
    ).build();
    final convoEntity = convoQuery.findFirst();
    convoQuery.close();
    
    if (convoEntity != null) {
      final msgQuery = ObjectBoxManager.messages.query(
        MessageEntity_.conversationId.equals(conversationId)
      ).build();
      ObjectBoxManager.messages.removeMany(msgQuery.findIds());
      msgQuery.close();
      
      ObjectBoxManager.conversations.remove(convoEntity.objectId);
    }
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
    } catch (e) {
      print('Failed to generate embedding: $e');
    }
    
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
    
    final convoQuery = ObjectBoxManager.conversations.query(
      ConversationEntity_.id.equals(conversationId)
    ).build();
    final convoEntity = convoQuery.findFirst();
    convoQuery.close();
    
    if (convoEntity != null) {
      convoEntity.updatedAt = DateTime.now().millisecondsSinceEpoch;
      ObjectBoxManager.conversations.put(convoEntity);
    }
    
    return Message.fromEntity(entity);
  }
  
List<Message> getMessages(String conversationId) {
  final query = ObjectBoxManager.messages.query(
    MessageEntity_.conversationId.equals(conversationId)
  ).build();
  
  final entities = query.find();
  query.close();
  
  entities.sort((a, b) => a.timestamp.compareTo(b.timestamp));
  return entities.map((e) => Message.fromEntity(e)).toList();
}


  Future<List<Message>> searchMessages({
    required String query,
    int limit = 10,
    String? conversationId,
  }) async {
    final queryResult = await _embeddingModel.generateEmbedding(text: query);
    final queryEmbedding = queryResult.embeddings;
    
    final entities = conversationId != null
        ? ObjectBoxManager.messages.query(
            MessageEntity_.conversationId.equals(conversationId)
          ).build().find()
        : ObjectBoxManager.messages.getAll();
    
    final messagesWithEmbeddings = entities.where((e) => e.embedding != null).toList();
    if (messagesWithEmbeddings.isEmpty) return [];
    
    final scored = messagesWithEmbeddings.map((entity) {
      final similarity = _cosineSimilarity(queryEmbedding, entity.embedding!);
      return {'entity': entity, 'similarity': similarity};
    }).toList();
    
    scored.sort((a, b) => (b['similarity'] as double).compareTo(a['similarity'] as double));
    
    return scored.take(limit)
        .map((item) => Message.fromEntity(item['entity'] as MessageEntity))
        .toList();
  }
  
  String getConversationContext(String conversationId, {int lastN = 5}) {
    final messages = getMessages(conversationId);
    final recentMessages = messages.length > lastN 
        ? messages.sublist(messages.length - lastN) 
        : messages;
    
    final buffer = StringBuffer();
    for (final msg in recentMessages) {
      final speaker = msg.isUser ? 'User' : 'Assistant';
      buffer.writeln('$speaker: ${msg.text}');
    }
    return buffer.toString();
  }
  
  List<ChatMessage> messagesToChatMessages(List<Message> messages) {
    return messages.map((msg) {
      return ChatMessage(
        content: msg.text,
        role: msg.isUser ? 'user' : 'assistant',
      );
    }).toList();
  }
  
  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return 0.0;
    
    double dotProduct = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;
    
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magnitudeA += a[i] * a[i];
      magnitudeB += b[i] * b[i];
    }
    
    if (magnitudeA == 0.0 || magnitudeB == 0.0) return 0.0;
    return dotProduct / (magnitudeA * magnitudeB);
  }
  
  Future<void> loadFromDisk() async {}
  
  void dispose() {
    ObjectBoxManager.close();
  }
}