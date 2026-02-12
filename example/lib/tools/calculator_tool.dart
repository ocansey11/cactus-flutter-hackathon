class CalculatorTool {
  static Map<String, dynamic> getDefinition() {
    return {
      'type': 'function',
      'function': {
        'name': 'calculate',
        'description': 'Perform basic math calculations',
        'parameters': {
          'type': 'object',
          'properties': {
            'operation': {
              'type': 'string',
              'enum': ['add', 'subtract', 'multiply', 'divide'],
              'description': 'The operation to perform',
            },
            'a': {
              'type': 'number',
              'description': 'First number',
            },
            'b': {
              'type': 'number',
              'description': 'Second number',
            },
          },
          'required': ['operation', 'a', 'b'],
        },
      },
    };
  }
  
  static Future<String> execute(Map<String, dynamic> arguments) async {
    final operation = arguments['operation'] as String;
    final a = (arguments['a'] as num).toDouble();
    final b = (arguments['b'] as num).toDouble();
    
    double result;
    switch (operation) {
      case 'add':
        result = a + b;
        break;
      case 'subtract':
        result = a - b;
        break;
      case 'multiply':
        result = a * b;
        break;
      case 'divide':
        if (b == 0) return 'Error: Division by zero';
        result = a / b;
        break;
      default:
        return 'Error: Unknown operation';
    }
    
    return '$a $operation $b = $result';
  }
}
