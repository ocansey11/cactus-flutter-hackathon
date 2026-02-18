/// Paper Analysis Prompts
/// Owns prompts for deep section-level analysis of research papers.
/// Each section type gets its own focused prompt so the model
/// knows exactly what to extract rather than guessing.
class PaperAnalysisPrompts {
  PaperAnalysisPrompts._();

  /// Routes to the correct section prompt by name.
  /// Falls back to [general] if section is unknown.
  static String forSection({
    required String paperTitle,
    required String section,
    required String context,
  }) {
    switch (section.toLowerCase()) {
      case 'methodology':
        return methodology(paperTitle: paperTitle, context: context);
      case 'findings':
      case 'results':
        return findings(paperTitle: paperTitle, context: context);
      case 'introduction':
      case 'background':
        return background(paperTitle: paperTitle, context: context);
      case 'conclusion':
      case 'conclusions':
        return conclusion(paperTitle: paperTitle, context: context);
      default:
        return general(paperTitle: paperTitle, context: context);
    }
  }

  static String methodology({
    required String paperTitle,
    required String context,
  }) =>
      '''Analyze the METHODOLOGY of "$paperTitle" using ONLY the content below.

Cover:
- Research design and approach
- Data sources and collection methods
- Tools, frameworks, or algorithms used
- Evaluation criteria

Content:
$context

Be specific. Quote or reference exact details from the content when relevant.''';

  static String findings({
    required String paperTitle,
    required String context,
  }) =>
      '''Analyze the FINDINGS AND RESULTS of "$paperTitle" using ONLY the content below.

Cover:
- Key results and metrics
- What was proven or disproven
- Comparisons to baselines or prior work
- Statistical significance where mentioned

Content:
$context

Be specific. Do not generalise beyond what the content states.''';

  static String background({
    required String paperTitle,
    required String context,
  }) =>
      '''Analyze the BACKGROUND AND INTRODUCTION of "$paperTitle" using ONLY the content below.

Cover:
- Problem being addressed
- Motivation and context
- Prior work referenced
- Gap the paper fills

Content:
$context''';

  static String conclusion({
    required String paperTitle,
    required String context,
  }) =>
      '''Analyze the CONCLUSIONS of "$paperTitle" using ONLY the content below.

Cover:
- Main takeaways
- Limitations acknowledged by the authors
- Future work suggested
- Real-world implications

Content:
$context''';

  static String general({
    required String paperTitle,
    required String context,
  }) =>
      '''Analyze the research paper "$paperTitle" using ONLY the content below.

Provide a structured analysis covering the most prominent sections present in the content. \
Use markdown headers for each section you identify.

Content:
$context''';
}
