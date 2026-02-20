import 'package:flutter/material.dart';
import '../services/function_gemma_downloader.dart';

class DownloadFunctionGemmaPage extends StatefulWidget {
  const DownloadFunctionGemmaPage({super.key});

  @override
  State<DownloadFunctionGemmaPage> createState() => _DownloadFunctionGemmaPageState();
}

class _DownloadFunctionGemmaPageState extends State<DownloadFunctionGemmaPage> {
  bool _isDownloading = false;
  bool _isDownloaded = false;
  double _progress = 0.0;
  String _statusMessage = 'Ready to download';

  @override
  void initState() {
    super.initState();
    _checkIfDownloaded();
  }

  Future<void> _checkIfDownloaded() async {
    final downloaded = await FunctionGemmaDownloader.isDownloaded();
    setState(() {
      _isDownloaded = downloaded;
      if (downloaded) {
        _statusMessage = 'FunctionGemma is already downloaded';
        _progress = 1.0;
      }
    });
  }

  Future<void> _downloadModel() async {
    setState(() {
      _isDownloading = true;
      _statusMessage = 'Starting download...';
      _progress = 0.0;
    });

    final success = await FunctionGemmaDownloader.download(
      onProgress: (progress, message) {
        setState(() {
          _progress = progress;
          _statusMessage = message;
        });
      },
    );

    setState(() {
      _isDownloading = false;
      _isDownloaded = success;
      if (success) {
        _statusMessage = 'Download complete! You can now use FunctionGemma';
      } else {
        _statusMessage = 'Download failed. Check the console for details.';
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Download FunctionGemma'),
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
              const Text(
                      'FunctionGemma for Hackathon',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'FunctionGemma is a specialized model fine-tuned for function calling and tool use. '
                      'It generates properly structured JSON for tool selection.',
                      style: TextStyle(fontSize: 14),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      '⚠️ Important: You need to find the actual GGUF download URL first!',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.orange,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'See FUNCTIONGEMMA_SETUP.md for instructions on finding the model.',
                      style: TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            if (_isDownloaded)
              Card(
                color: Colors.green[50],
                child: const Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      Icon(Icons.check_circle, color: Colors.green, size: 32),
                      SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'FunctionGemma is ready to use!',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.green,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              Column(
                children: [
                  ElevatedButton(
                    onPressed: _isDownloading ? null : _downloadModel,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      backgroundColor: Colors.deepPurple,
                    ),
                    child: _isDownloading
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Text(
                            'Download FunctionGemma',
                            style: TextStyle(fontSize: 16),
                          ),
                  ),
                  const SizedBox(height: 16),
                  if (_isDownloading || _progress > 0)
                    Column(
                      children: [
                        LinearProgressIndicator(value: _progress),
                        const SizedBox(height: 8),
                        Text(
                          _statusMessage,
                          style: const TextStyle(fontSize: 12),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                ],
              ),
            const SizedBox(height: 24),
            const Divider(),
            const SizedBox(height: 16),
            const Text(
              'Model Information',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            _buildInfoRow('Model Size', '~1-2GB (quantized)'),
            _buildInfoRow('Format', 'GGUF'),
            _buildInfoRow('Specialization', 'Function calling & tool use'),
            _buildInfoRow('Quantization', 'Q4_K_M or Q8_0 recommended'),
            const SizedBox(height: 24),
            const Text(
              'Download Options',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            ...FunctionGemmaDownloader.getDownloadOptions().map(
              (option) => Card(
                child: ListTile(
                  title: Text(option['name']!),
                  subtitle: Text('Size: ${option['size']}, ${option['quantization']}'),
                  trailing: const Icon(Icons.info_outline),
                  dense: true,
                ),
              ),
            ),
            const Spacer(),
            Text(
              'Note: After downloading, update main.dart to use "function-gemma-2b" as the model.',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[600],
                fontStyle: FontStyle.italic,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            flex: 2,
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
          Expanded(
            flex: 3,
            child: Text(value),
          ),
        ],
      ),
    );
  }
}
