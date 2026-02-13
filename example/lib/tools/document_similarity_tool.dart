import 'dart:math';

class DocumentSimilarityTool {
  static Map<String, dynamic> getDefinition() {
    return {
      'type': 'function',
      'function': {
        'name': 'compute_document_similarity',
        'description': 'Analyze similarity relationships between documents in the knowledge base',
        'parameters': {
          'type': 'object',
          'properties': {
            'threshold': {
              'type': 'number',
              'description': 'Minimum similarity score to include (0.0-1.0)',
              'default': 0.5,
            },
          },
          'required': [],
        },
      },
    };
  }
  
  static Future<String> execute(Map<String, dynamic> arguments) async {
    final threshold = (arguments['threshold'] as num?)?.toDouble() ?? 0.5;
    
    return 'Document similarity analysis:\n'
        'Total documents: 0\n'
        'Threshold: $threshold\n'
        'No documents in knowledge base yet.';
  }
  
  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      throw ArgumentError('Vectors must have same dimensions');
    }
    
    double dotProduct = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;
    
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magnitudeA += a[i] * a[i];
      magnitudeB += b[i] * b[i];
    }
    
    magnitudeA = sqrt(magnitudeA);
    magnitudeB = sqrt(magnitudeB);
    
    if (magnitudeA == 0.0 || magnitudeB == 0.0) {
      return 0.0;
    }
    
    return dotProduct / (magnitudeA * magnitudeB);
  }
}
