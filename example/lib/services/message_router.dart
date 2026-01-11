import 'package:cactus/cactus.dart';

/// Represents the type of processing to apply to a query
enum MessageType {
  /// Use stored documents with RAG
  rag,

  /// Simple LLM conversation without documents
  simpleChat,

  /// Future: Tool/function calling
  toolCalling,
}

/// Result of routing analysis
class RouterResult {
  final MessageType messageType;
  final bool isRelevant;
  final double? relevanceScore;
  final String? reason;

  RouterResult({
    required this.messageType,
    this.isRelevant = false,
    this.relevanceScore,
    this.reason,
  });

  @override
  String toString() => 'RouterResult(type: $messageType, relevant: $isRelevant, score: $relevanceScore, reason: $reason)';
}

/// Systematically determines which processing path to use for a query
class MessageRouter {
  final CactusRAG rag;

  // Configuration
  static const double relevanceThreshold = 0.5;
  static const int searchLimit = 3;

  MessageRouter({required this.rag});

  /// Main entry point: analyzes query and determines processing type
  Future<RouterResult> route({
    required String query,
    required bool hasPendingDocs,
  }) async {
    try {
      // Step 1: If user uploaded new documents, always use RAG
      if (hasPendingDocs) {
        return RouterResult(
          messageType: MessageType.rag,
          isRelevant: true,
          reason: 'User has pending documents to process',
        );
      }

      // Step 2: Check if we have documents in the database
      final existingDocs = await rag.getAllDocuments();

      if (existingDocs.isEmpty) {
        // No documents at all - fall back to simple chat
        return RouterResult(
          messageType: MessageType.simpleChat,
          isRelevant: false,
          reason: 'No documents in database',
        );
      }

      // Step 3: We have documents - check query relevance
      return await _checkRelevance(query);
    } catch (e) {
      // On error, default to simple chat
      print('Error in router: $e');
      return RouterResult(
        messageType: MessageType.simpleChat,
        isRelevant: false,
        reason: 'Error during routing: $e',
      );
    }
  }

  /// Checks if query is relevant to stored documents using similarity search
  Future<RouterResult> _checkRelevance(String query) async {
    try {
      final results = await rag.search(text: query, limit: searchLimit);

      // No results found
      if (results.isEmpty) {
        return RouterResult(
          messageType: MessageType.simpleChat,
          isRelevant: false,
          relevanceScore: 1.0, // Max distance = no relevance
          reason: 'No search results for query',
        );
      }

      final topDistance = results.first.distance;

      // Check if top result meets threshold
      if (topDistance < relevanceThreshold) {
        return RouterResult(
          messageType: MessageType.rag,
          isRelevant: true,
          relevanceScore: topDistance,
          reason: 'Query is relevant to documents (distance: $topDistance < $relevanceThreshold)',
        );
      } else {
        return RouterResult(
          messageType: MessageType.simpleChat,
          isRelevant: false,
          relevanceScore: topDistance,
          reason: 'Query not relevant to documents (distance: $topDistance >= $relevanceThreshold)',
        );
      }
    } catch (e) {
      print('Error checking relevance: $e');
      return RouterResult(
        messageType: MessageType.simpleChat,
        isRelevant: false,
        reason: 'Error during relevance check: $e',
      );
    }
  }

  /// Gets a human-readable explanation of the routing decision
  String getExplanation(RouterResult result) {
    return switch (result.messageType) {
      MessageType.rag => result.isRelevant
          ? '📚 Using RAG - Your question is relevant to the uploaded documents'
          : '📚 Using RAG - Processing with document context',
      MessageType.simpleChat => '💬 Using simple chat - No relevant documents found',
      MessageType.toolCalling => '🔧 Tool calling mode (not yet implemented)',
    };
  }
}
