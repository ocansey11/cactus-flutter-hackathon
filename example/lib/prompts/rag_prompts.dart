/// RAG Prompts
/// Owns all prompts related to document grounding and context injection.
/// The system prompt is kept separate from the user prompt so ChatService
/// can pass them independently to the model.
class RAGPrompts {
  RAGPrompts._();

  /// System prompt injected as the [system] role.
  /// Keeps the model grounded to the provided context only.
  static String system() => '''You are JarvisOS Research Assistant — a precise, \
privacy-first AI running entirely on-device.

Your job is to answer questions using ONLY the document context provided to you. \
You do not speculate. You do not use training knowledge to fill gaps. \
If the answer is not in the context, say exactly: \
"I cannot find that information in the provided documents."

Be concise. Cite which document a fact comes from when possible.''';

  /// User turn prompt. Injects retrieved chunks + the user query.
  /// [context] is the joined chunk text from RAGService.buildContext().
  /// [query] is the raw user question.
  static String user({required String query, required String context}) => '''DOCUMENT CONTEXT:
---
$context
---

Question: $query

Answer using ONLY the document context above:''';

  /// Variant for when a tool result is also available alongside RAG context.
  static String userWithToolResult({
    required String query,
    required String context,
    required String toolResult,
  }) =>
      '''TOOL OUTPUT:
$toolResult

DOCUMENT CONTEXT:
---
$context
---

Question: $query

Use the tool output and document context above to answer:''';
}
