import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'paper_viewer_page.dart';

class PapersPage extends StatefulWidget {
  final ProjectService projectService;

  const PapersPage({
    super.key,
    required this.projectService,
  });

  @override
  State<PapersPage> createState() => _PapersPageState();
}

class _PapersPageState extends State<PapersPage> {
  List<String> _papers = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPapers();
  }

  Future<void> _loadPapers() async {
    if (widget.projectService.currentProject == null) {
      setState(() => _isLoading = false);
      return;
    }

    try {
      final papers = await widget.projectService.storageService.listPapers(
        widget.projectService.currentProject!.name,
      );
      
      setState(() {
        _papers = papers;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading papers: $e')),
        );
      }
    }
  }

  String _formatFileSize(String fileName) {
    return fileName;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          '${widget.projectService.currentProject?.name ?? 'Project'} - Papers',
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _papers.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.description_outlined,
                        size: 64,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'No papers yet',
                        style: TextStyle(
                          fontSize: 18,
                          color: Colors.grey[600],
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Upload papers from the chat page',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey[500],
                        ),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  itemCount: _papers.length,
                  itemBuilder: (context, index) {
                    final paper = _papers[index];
                    final extension = paper.split('.').last.toUpperCase();
                    
                    return ListTile(
                      leading: CircleAvatar(
                        backgroundColor: _getColorForExtension(extension),
                        child: Text(
                          extension,
                          style: const TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      title: Text(paper),
                      subtitle: Text(
                        'Added to project',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[600],
                        ),
                      ),
                      trailing: IconButton(
                        icon: const Icon(Icons.more_vert),
                        onPressed: () => _showPaperOptions(paper),
                      ),
                      onTap: () => _openPaper(paper),
                    );
                  },
                ),
    );
  }

  Color _getColorForExtension(String extension) {
    switch (extension.toLowerCase()) {
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

  void _showPaperOptions(String paper) {
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
                _openPaper(paper);
              },
            ),
            ListTile(
              leading: const Icon(Icons.delete_outline, color: Colors.red),
              title: const Text(
                'Delete',
                style: TextStyle(color: Colors.red),
              ),
              onTap: () {
                Navigator.pop(context);
                _confirmDelete(paper);
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _openPaper(String paper) async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PaperViewerPage(
          fileName: paper,
          projectService: widget.projectService,
        ),
      ),
    );
  }

  void _confirmDelete(String paper) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Paper'),
        content: Text('Are you sure you want to delete "$paper"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _deletePaper(paper);
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  Future<void> _deletePaper(String paper) async {
    try {
      final projectName = widget.projectService.currentProject!.name;
      final projectId = widget.projectService.currentProject!.id;

      // Delete the physical file
      final deleted = await widget.projectService.storageService.deletePaper(
        projectName: projectName,
        fileName: paper,
      );

      if (!deleted) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Paper file not found'),
              backgroundColor: Colors.orange,
            ),
          );
        }
        return;
      }

      // Remove the document mapping from the project
      await widget.projectService.removeDocumentFromProject(
        projectId: projectId,
        documentFileName: paper,
      );

      // Reload the papers list
      await _loadPapers();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Deleted: $paper'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error deleting paper: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
}
