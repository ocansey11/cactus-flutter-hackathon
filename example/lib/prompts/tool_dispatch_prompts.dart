/// Tool Dispatch Prompts
/// Owns both stages of function calling:
///   Stage 1 — Qwen simplifies the raw user query into a compact intent string.
///   Stage 2 — Gemma maps the intent to a structured JSON function call.
///
/// Each prompt is versioned via a comment so we can track what changed and why.
class ToolDispatchPrompts {
  ToolDispatchPrompts._();

  // ─── Stage 1: Query Simplification (Qwen) ────────────────────────────────

  /// v2 — added confidence signal, preserved paper titles explicitly,
  /// removed "show project statistics" route (now keyword-matched upstream).
  static String querySimplify({required String query}) => '''You are a precise intent extractor. \
Output ONE line only. No explanations. No thinking.

Rules:
- Preserve paper titles EXACTLY as written — never shorten them
- Output must start with one of: create / analyze / show / none
- If no tool is needed, output exactly: none

Available intents:
  create summary note about <paper title>
  create objective note: <content>
  create plan note: <content>
  create general note: <content>
  analyze <section> section of <paper title>
  show project papers
  show project notes
  show project statistics
  none

Input: "$query"
Output:''';

  // ─── Stage 2: Tool Matching (Gemma) ──────────────────────────────────────

  /// v2 — added confidence field (0.0–1.0), added "none" function for
  /// explicit no-op signal instead of relying on null parsing.
  /// Confidence < 0.7 should be treated as no-tool by the caller.
  static String toolMatch({required String intent}) => '''Map the intent below to a JSON function call. \
Output ONLY the JSON. No markdown. No explanation.

Functions:
  create_project_note  → params: note_type (summary|objective|plan|general), paper_name?, content?
  analyze_research_paper → params: paper_name, section (methodology|findings|introduction|conclusion|background)
  get_project_context  → params: info_type (statistics|papers|notes|all)
  none                 → params: {}

Rules:
- confidence: how certain you are this is the right function (0.0 to 1.0)
- Use "none" if the intent does not clearly map to a function
- Extract ACTUAL values, never use placeholder text

Examples:
Intent: "create summary note about Disability Fairness paper"
{"function":"create_project_note","parameters":{"note_type":"summary","paper_name":"Disability Fairness paper"},"confidence":0.95}

Intent: "analyze methodology section of AI Bias in Hiring"
{"function":"analyze_research_paper","parameters":{"paper_name":"AI Bias in Hiring","section":"methodology"},"confidence":0.92}

Intent: "none"
{"function":"none","parameters":{},"confidence":1.0}

Intent: "$intent"
''';
}
