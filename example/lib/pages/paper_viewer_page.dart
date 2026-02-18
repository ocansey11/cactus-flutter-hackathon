import 'package:flutter/material.dart';
import '../services/project_service.dart';
import '../services/rag_service.dart';

class PaperViewerPage extends StatefulWidget {
  final String fileName;
  final ProjectService projectService;
  final RAGService? ragService;

  const PaperViewerPage({
    super.key,
    required this.fileName,
    required this.projectService,
    this.ragService,
  });

  @override
  State<PaperViewerPage> createState() => _PaperViewerPageState();
}

class _PaperViewerPageState extends State<PaperViewerPage> {
  bool _isLoading = true;
  String _content = '';
  String _error = '';

  @override
  void initState() {
    super.initState();
    _loadContent();
  }

  Future<void> _loadContent() async {
    if (widget.ragService == null) {
      setState(() {
        _error = 'RAG service not available.';
        _isLoading = false;
      });
      return;
    }

    try {
      final docs = await widget.ragService!.getAllDocuments();
      final doc = docs.firstWhere(
        (d) => d.fileName == widget.fileName,
        orElse: () => throw Exception('Document not found in knowledge base.'),
      );

      final content = doc.chunks.map((c) => c.content).join('\n\n');

      setState(() {
        _content = content.isEmpty ? 'No content available.' : content;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  void _showSearch() {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Search in Document'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: 'Search term',
            border: OutlineInputBorder(),
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final term = controller.text.toLowerCase();
              if (term.isNotEmpty) {
                final count =
                    _content.toLowerCase().split(term).length - 1;
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                      content:
                          Text('Found "$term" $count time(s)')),
                );
              }
            },
            child: const Text('Search'),
          ),
        ],
      ),
    );
  }

  void _showStats() {
    final words =
        _content.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).length;
    final lines = _content.split('\n').length;
    final paragraphs = _content.split(RegExp(r'\n\s*\n')).length;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Document Statistics'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _StatRow(label: 'Characters', value: _content.length.toString()),
            _StatRow(label: 'Words', value: words.toString()),
            _StatRow(label: 'Lines', value: lines.toString()),
            _StatRow(label: 'Paragraphs', value: paragraphs.toString()),
            _StatRow(
              label: 'Est. reading time',
              value: '${(words / 200).ceil()} min',
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.fileName),
        actions: [
          if (_content.isNotEmpty) ...[
            IconButton(
              icon: const Icon(Icons.search),
              onPressed: _showSearch,
              tooltip: 'Search',
            ),
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showStats,
              tooltip: 'Stats',
            ),
          ],
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error.isNotEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.error_outline,
                          size: 64, color: Colors.red[300]),
                      const SizedBox(height: 16),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 32),
                        child: Text(_error,
                            textAlign: TextAlign.center,
                            style: const TextStyle(fontSize: 16)),
                      ),
                    ],
                  ),
                )
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: SelectableText(
                    _content,
                    style: const TextStyle(fontSize: 14, height: 1.6),
                  ),
                ),
    );
  }
}

class _StatRow extends StatelessWidget {
  final String label;
  final String value;

  const _StatRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(value, style: TextStyle(color: Colors.grey[700])),
        ],
      ),
    );
  }
}
