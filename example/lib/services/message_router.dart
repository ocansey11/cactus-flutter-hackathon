import 'package:cactus/cactus.dart';
import 'function_calling_service.dart';

enum MessageType {
  rag,
  simpleChat,
  toolCalling,
}

class RouterResult {
  final MessageType messageType;
  final bool isRelevant;
  final double? relevanceScore;
  final String? reason;
  final FunctionCallResult? functionResult;

  RouterResult({
    required this.messageType,
    this.isRelevant = false,
    this.relevanceScore,
    this.reason,
    this.functionResult,
  });

  @override
  String toString() => 'RouterResult(type: $messageType, relevant: $isRelevant, score: $relevanceScore, reason: $reason, function: $functionResult)';
}

class MessageRouter {
  final CactusRAG rag;
  final FunctionCallingService? functionService;

  static const double relevanceThreshold = 0.5;
  static const int searchLimit = 3;

  MessageRouter({
    required this.rag,
    this.functionService,
  });

  Future<RouterResult> route({
    required String query,
    required bool hasPendingDocs,
  }) async {
    
    FunctionCallResult? functionResult;

    try {
      if (functionService != null) {
        functionResult = await functionService!.analyzeQuery(query);
      }
    } catch (e) {
      functionResult = null;
    }

    try {
      if (hasPendingDocs) {
        return RouterResult(
          messageType: MessageType.rag,
          isRelevant: true,
          reason: 'User has pending documents to process',
          functionResult: functionResult,
        );
      }

      final existingDocs = await rag.getAllDocuments();

      if (existingDocs.isEmpty) {
        return RouterResult(
          messageType: MessageType.simpleChat,
          isRelevant: false,
          reason: 'No documents in database',
          functionResult: functionResult,
        );
      }

      return await _checkRelevance(query, functionResult);
    } catch (e) {
      return RouterResult(
        messageType: MessageType.simpleChat,
        isRelevant: false,
        reason: 'Error during routing: $e',
        functionResult: functionResult,
      );
    }
  }

  Future<RouterResult> _checkRelevance(
    String query,
    FunctionCallResult? functionResult,
  ) async {
    try {
      final results = await rag.search(text: query, limit: searchLimit);
      
      if (results.isEmpty) {
        return RouterResult(
          messageType: MessageType.simpleChat,
          isRelevant: false,
          relevanceScore: 1.0,
          reason: 'No search results for query',
          functionResult: functionResult,
        );
      }

      final topDistance = results.first.distance;
      
      if (topDistance < relevanceThreshold) {
        return RouterResult(
          messageType: MessageType.rag,
          isRelevant: true,
          relevanceScore: topDistance,
          reason: 'Query is relevant to documents (distance: $topDistance < $relevanceThreshold)',
          functionResult: functionResult,
        );
      } else {
        return RouterResult(
          messageType: MessageType.simpleChat,
          isRelevant: false,
          relevanceScore: topDistance,
          reason: 'Query not relevant to documents (distance: $topDistance >= $relevanceThreshold)',
          functionResult: functionResult,
        );
      }
    } catch (e) {
      return RouterResult(
        messageType: MessageType.simpleChat,
        isRelevant: false,
        reason: 'Error during relevance check: $e',
        functionResult: functionResult,
      );
    }
  }

  String getExplanation(RouterResult result) {
    return switch (result.messageType) {
      MessageType.rag => result.isRelevant
          ? 'Using RAG - Your question is relevant to the uploaded documents'
          : 'Using RAG - Processing with document context',
      MessageType.simpleChat => 'Using simple chat - No relevant documents found',
      MessageType.toolCalling => 'Tool calling mode (not yet implemented)',
    };
  }
}
