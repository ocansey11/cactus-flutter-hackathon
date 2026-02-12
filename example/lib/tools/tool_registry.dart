import 'weather_tool.dart';
import 'calculator_tool.dart';

class ToolRegistry {
  static final Map<String, Function> _tools = {
    'get_weather': WeatherTool.execute,
    'calculate': CalculatorTool.execute,
  };
  
  static List<Map<String, dynamic>> getAllDefinitions() {
    return [
      WeatherTool.getDefinition(),
      CalculatorTool.getDefinition(),
    ];
  }
  
  static Future<String> executeTool(String name, Map<String, dynamic> arguments) async {
    final tool = _tools[name];
    if (tool == null) {
      throw Exception('Tool not found: $name');
    }
    return await tool(arguments);
  }
  
  static bool hasTool(String name) {
    return _tools.containsKey(name);
  }
}
