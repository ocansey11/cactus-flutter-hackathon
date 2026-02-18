import 'dart:io';
import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import 'package:pdfx/pdfx.dart';

class PaperViewerPage extends StatefulWidget {
  final String fileName;
  final ProjectService projectService;

  const PaperViewerPage({
    super.key,
    required this.fileName,
    required this.projectService,
  });

  @override
  State<PaperViewerPage> createState() => _PaperViewerPageState();
}

class _PaperViewerPageState extends State<PaperViewerPage> {
  bool _isLoading = true;
  String _textContent = '';
  String _errorMessage = '';
  PdfController? _pdfController;
  File? _paperFile;
  bool _isPdf = false;

  @override
  void initState() {
    super.initState();
    _loadPaper();
  }

  @override
  void dispose() {
    _pdfController?.dispose();
    super.dispose();
  }

  Future<void> _loadPaper() async {
    try {
      final projectName = widget.projectService.currentProject!.name;
      final paperFile = await widget.projectService.storageService.getPaperFile(
        projectName: projectName,
        fileName: widget.fileName,
      );

      if (paperFile == null) {
        setState(() {
          _isLoading = false;
          _errorMessage = 'Paper file not found';
        });
        return;
      }

      _paperFile = paperFile;
      final extension = widget.fileName.split('.').last.toLowerCase();
      _isPdf = extension == 'pdf';

      if (_isPdf) {
        // Load PDF with native viewer
        _pdfController = PdfController(
          document: PdfDocument.openFile(paperFile.path),
        );
        setState(() {
          _isLoading = false;
        });
      } else {
        // Load text content for TXT/MD files
        final content = await paperFile.readAsString();
        setState(() {
          _textContent = content;
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Error loading paper: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.fileName),
        actions: [
          if (!_isPdf && _textContent.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.search),
              onPressed: _showSearchDialog,
              tooltip: 'Search in document',
            ),
          if (_textContent.isNotEmpty && !_isPdf)
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showStats,
              tooltip: 'Document stats',
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _errorMessage.isNotEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.error_outline,
                        size: 64,
                        color: Colors.red[300],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _errorMessage,
                        style: const TextStyle(fontSize: 16),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                )
              : _isPdf
                  ? PdfView(
                      controller: _pdfController!,
                      scrollDirection: Axis.vertical,
                      builders: PdfViewBuilders<DefaultBuilderOptions>(
                        options: const DefaultBuilderOptions(),
                        documentLoaderBuilder: (_) =>
                            const Center(child: CircularProgressIndicator()),
                        pageLoaderBuilder: (_) =>
                            const Center(child: CircularProgressIndicator()),
                        errorBuilder: (_, error) => Center(
                          child: Text('Error loading PDF: $error'),
                        ),
                      ),
                    )
                  : SingleChildScrollView(
                      padding: const EdgeInsets.all(16),
                      child: SelectableText(
                        _textContent,
                        style: const TextStyle(
                          fontSize: 14,
                          height: 1.5,
                        ),
                      ),
                    ),
    );
  }

  void _showSearchDialog() {
    final searchController = TextEditingController();
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Search in Document'),
        content: TextField(
          controller: searchController,
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
              final term = searchController.text.toLowerCase();
              if (term.isNotEmpty) {
                final occurrences = _textContent.toLowerCase().split(term).length - 1;
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('Found "$term" $occurrences time(s)'),
                  ),
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
    final words = _textContent.split(RegExp(r'\s+'));
    final wordCount = words.where((w) => w.isNotEmpty).length;
    final charCount = _textContent.length;
    final lines = _textContent.split('\n').length;
    final paragraphs = _textContent.split(RegExp(r'\n\s*\n')).length;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Document Statistics'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _StatRow(label: 'Characters', value: charCount.toString()),
            _StatRow(label: 'Words', value: wordCount.toString()),
            _StatRow(label: 'Lines', value: lines.toString()),
            _StatRow(label: 'Paragraphs', value: paragraphs.toString()),
            const SizedBox(height: 8),
            _StatRow(
              label: 'Estimated reading time',
              value: '${(wordCount / 200).ceil()} min',
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
}

class _StatRow extends StatelessWidget {
  final String label;
  final String value;

  const _StatRow({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
          Text(
            value,
            style: TextStyle(color: Colors.grey[700]),
          ),
        ],
      ),
    );
  }
}
