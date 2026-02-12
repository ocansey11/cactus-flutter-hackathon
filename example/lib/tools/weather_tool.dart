class WeatherTool {
  static Map<String, dynamic> getDefinition() {
    return {
      'type': 'function',
      'function': {
        'name': 'get_weather',
        'description': 'Get current weather for a location',
        'parameters': {
          'type': 'object',
          'properties': {
            'location': {
              'type': 'string',
              'description': 'City name, e.g. San Francisco',
            },
            'unit': {
              'type': 'string',
              'enum': ['celsius', 'fahrenheit'],
              'description': 'Temperature unit',
            },
          },
          'required': ['location'],
        },
      },
    };
  }
  
  static Future<String> execute(Map<String, dynamic> arguments) async {
    final location = arguments['location'] as String;
    final unit = arguments['unit'] as String? ?? 'celsius';
    
    await Future.delayed(const Duration(milliseconds: 500));
    
    return 'Weather in $location: 22°${unit == 'celsius' ? 'C' : 'F'}, Sunny';
  }
}
