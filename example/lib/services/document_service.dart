import 'dart:io';
import 'package:read_pdf_text/read_pdf_text.dart';

class DocumentService {
  static Future<String> extractPdfText(String filePath) async {
    String content = await ReadPdfText.getPDFtext(filePath);
    content = content.replaceAll('a', '"');
    content = content.replaceAll('ʼ', "'");
    return content;
  }
  
  static Future<String> readTextFile(String filePath) async {
    final file = File(filePath);
    return await file.readAsString();
  }
  
  static Future<String> extractContent(String filePath, String extension) async {
    if (extension == 'pdf') {
      return await extractPdfText(filePath);
    } else {
      return await readTextFile(filePath);
    }
  }
  
  static bool isValidContent(String content) {
    return content.trim().isNotEmpty;
  }
  
  static Map<String, dynamic> createDocumentMetadata({
    required String fileName,
    required String filePath,
    required String content,
    required int fileSize,
  }) {
    return {
      'fileName': fileName,
      'filePath': filePath,
      'content': content,
      'fileSize': fileSize,
    };
  }
}
