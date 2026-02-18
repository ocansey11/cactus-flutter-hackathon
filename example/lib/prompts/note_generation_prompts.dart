/// Note Generation Prompts
/// Owns prompts for all AI-generated note types:
/// summary, objective, plan, general.
/// Each method produces a self-contained user-turn prompt.
class NoteGenerationPrompts {
  NoteGenerationPrompts._();

  /// Generates a structured research paper summary.
  /// [paperTitle] is the document name. [context] is the RAG-retrieved text.
  static String summary({
    required String paperTitle,
    required String context,
  }) =>
      '''Summarize the research paper "$paperTitle" using ONLY the content below.

Structure your response in markdown with these exact sections:
## Overview
## Key Objectives
## Methodology
## Findings
## Conclusions

Content:
$context

Write clearly and concisely. Do not invent details not present in the content above.''';

  /// Formats a user-written project objective into structured markdown.
  static String objective({required String userContent}) => '''Format the following \
project objective into structured markdown.

Use these exact sections:
## Goals
## Background & Motivation
## Expected Outcomes

Raw input:
$userContent

Keep the original meaning intact. Add structure, not new content.''';

  /// Formats a user-written project plan into structured markdown.
  static String plan({required String userContent}) => '''Format the following \
project plan into structured markdown.

Use these exact sections:
## Overview
## Phases
## Milestones
## Success Criteria

Raw input:
$userContent

Keep the original meaning intact. Add structure, not new content.''';
}
