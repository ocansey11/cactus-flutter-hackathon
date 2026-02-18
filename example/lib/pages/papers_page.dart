import 'package:flutter/material.dart';
import '../services/project_service.dart';
import '../services/rag_service.dart';
import 'paper_viewer_page.dart';

class PapersPage extends StatefulWidget {
  final ProjectService projectService;
  final RAGService? ragService;

  const PapersPage({
    super.key,
    required this.projectService,
    this.ragService,
  });

  @override
  State<PapersPage> createState() => _PapersPageState();
}

class _PapersPageState extends State<PapersPage> {
  @override
  Widget build(BuildContext context) {
    final project = widget.projectService.currentProject;

    if (project == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Papers')),
        body: const Center(child: Text('No active project.')),
      );
    }

    final papers = widget.projectService.getDocuments(project.id);

    return Scaffold(
      appBar: AppBar(title: Text('${project.name} — Papers')),
      body: papers.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.description_outlined,
                      size: 64, color: Colors.grey[400]),
                  const SizedBox(height: 16),
                  Text('No papers yet',
                      style: TextStyle(fontSize: 18, color: Colors.grey[600])),
                  const SizedBox(height: 8),
                  Text('Upload papers from the chat page',
                      style:
                          TextStyle(fontSize: 14, color: Colors.grey[500])),
                ],
              ),
            )
          : ListView.builder(
              itemCount: papers.length,
              itemBuilder: (context, index) {
                final paper = papers[index];
                final extension =
                    paper.fileName.split('.').last.toUpperCase();

                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: _colorForExtension(extension),
                    child: Text(
                      extension.length > 3
                          ? extension.substring(0, 3)
                          : extension,
                      style: const TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: Colors.white),
                    ),
                  ),
                  title: Text(paper.fileName),
                  subtitle: Text(
                    _formatSize(paper.fileSize),
                    style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                  ),
                  trailing: IconButton(
                    icon: const Icon(Icons.more_vert),
                    onPressed: () => _showOptions(paper.fileName, paper.id),
                  ),
                  onTap: () => _openPaper(paper.fileName),
                );
              },
            ),
    );
  }

  Color _colorForExtension(String ext) {
    switch (ext.toLowerCase()) {
      case 'pdf':
        return Colors.red;
      case 'txt':
        return Colors.blue;
      case 'md':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  String _formatSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  void _showOptions(String fileName, String docId) {
    showModalBottomSheet(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.open_in_new),
              title: const Text('Open'),
              onTap: () {
                Navigator.pop(context);
                _openPaper(fileName);
              },
            ),
            ListTile(
              leading: const Icon(Icons.delete_outline, color: Colors.red),
              title: const Text('Delete',
                  style: TextStyle(color: Colors.red)),
              onTap: () {
                Navigator.pop(context);
                _confirmDelete(fileName, docId);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _openPaper(String fileName) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PaperViewerPage(
          fileName: fileName,
          projectService: widget.projectService,
          ragService: widget.ragService,
        ),
      ),
    );
  }

  void _confirmDelete(String fileName, String docId) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Paper'),
        content: Text('Remove "$fileName" from this project?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              widget.projectService.deleteDocument(docId);
              setState(() {});
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}
