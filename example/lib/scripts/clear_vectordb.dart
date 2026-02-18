import 'package:cactus/cactus.dart';

Future<void> main() async {
  print('Initializing CactusRAG...');
  final rag = CactusRAG();
  
  await rag.initialize();
  
  print('Clearing vector database...');
  await rag.clearDatabase();
  
  print('Vector database cleared successfully!');
  print('The database has been reset and is ready for fresh data.');
  
  await rag.close();
}
