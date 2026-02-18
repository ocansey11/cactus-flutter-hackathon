import 'package:flutter/material.dart';
import '../models/document_graph.dart';
import 'dart:math' as math;

class DocumentGraphPage extends StatefulWidget {
  final DocumentGraph graphData;

  const DocumentGraphPage({
    super.key,
    required this.graphData,
  });

  @override
  State<DocumentGraphPage> createState() => _DocumentGraphPageState();
}

class _DocumentGraphPageState extends State<DocumentGraphPage> {
  late Map<String, Offset> nodePositions;
  late TransformationController transformationController;

  @override
  void initState() {
    super.initState();
    transformationController = TransformationController();
    _calculateNodePositions();
  }

  @override
  void dispose() {
    transformationController.dispose();
    super.dispose();
  }

  void _calculateNodePositions() {
    print('Calculating positions for ${widget.graphData.nodes.length} nodes');
    
    nodePositions = {};
    final nodes = widget.graphData.nodes;
    final random = math.Random(42); // Fixed seed for consistent layout
    
    // Random scatter within a contained area
    final scatterWidth = 1000.0;   // Total width for scattering
    final scatterHeight = 800.0;   // Total height for scattering
    final minDistance = 120.0;     // Minimum distance between nodes
    
    for (int i = 0; i < nodes.length; i++) {
      final node = nodes[i];
      bool positionFound = false;
      int attempts = 0;
      Offset? newPosition;
      
      // Try to find a position that doesn't overlap with existing nodes
      while (!positionFound && attempts < 100) {
        final x = 200 + random.nextDouble() * scatterWidth;
        final y = 100 + random.nextDouble() * scatterHeight;
        newPosition = Offset(x, y);
        
        // Check distance from all existing nodes
        bool tooClose = false;
        for (final existingPos in nodePositions.values) {
          final distance = (newPosition - existingPos).distance;
          if (distance < minDistance) {
            tooClose = true;
            break;
          }
        }
        
        if (!tooClose) {
          positionFound = true;
        }
        attempts++;
      }
      
      // If we couldn't find a good position after 100 attempts, use the last one anyway
      if (newPosition != null) {
        nodePositions[node.id] = newPosition;
        print('Node ${node.fileName} at (${newPosition.dx}, ${newPosition.dy})');
      }
    }
  }

  Color _getSimilarityColor(double similarity) {
    // Color gradient from blue (low) to green (high)
    if (similarity >= 0.8) return Colors.green;
    if (similarity >= 0.6) return Colors.lightGreen;
    if (similarity >= 0.4) return Colors.orange;
    return Colors.blue;
  }

  double _getSimilarityWidth(double similarity) {
    // Edge thickness based on similarity
    return 1.0 + (similarity * 4.0);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('Document Similarity Graph'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 1,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Graph Information'),
                  content: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _legendItem(Colors.grey.shade600, 'Connected', 'Documents with 80-100% similarity'),
                      _legendItem(Colors.grey.shade400, 'Isolated', 'Documents with no strong connections'),
                      const SizedBox(height: 16),
                      Text(
                        'Only showing connections ≥ ${(widget.graphData.threshold * 100).toStringAsFixed(0)}% similarity',
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Total documents: ${widget.graphData.nodes.length}',
                        style: const TextStyle(fontSize: 12),
                      ),
                      Text(
                        'Strong connections: ${widget.graphData.edges.length}',
                        style: const TextStyle(fontSize: 12),
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
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: InteractiveViewer(
              transformationController: transformationController,
              boundaryMargin: const EdgeInsets.all(double.infinity),
              minScale: 0.1,
              maxScale: 5.0,
              constrained: false,
              child: CustomPaint(
                size: const Size(1600, 1200),
                painter: GraphPainter(
                  nodes: widget.graphData.nodes,
                  edges: widget.graphData.edges,
                  nodePositions: nodePositions,
                ),
                child: const SizedBox(width: 1600, height: 1200),
              ),
            ),
          ),
          _buildControls(),
        ],
      ),
    );
  }

  Widget _legendItem(Color color, String range, String label) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 4,
            color: color,
          ),
          const SizedBox(width: 12),
          Text('$range - $label'),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(
          top: BorderSide(color: Colors.grey.shade300),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _controlButton(
            icon: Icons.zoom_in,
            label: 'Zoom In',
            onPressed: () {
              final currentScale = transformationController.value.getMaxScaleOnAxis();
              transformationController.value = Matrix4.identity()..scale(currentScale * 1.2);
            },
          ),
          _controlButton(
            icon: Icons.zoom_out,
            label: 'Zoom Out',
            onPressed: () {
              final currentScale = transformationController.value.getMaxScaleOnAxis();
              transformationController.value = Matrix4.identity()..scale(currentScale * 0.8);
            },
          ),
          _controlButton(
            icon: Icons.center_focus_strong,
            label: 'Reset',
            onPressed: () {
              transformationController.value = Matrix4.identity();
            },
          ),
        ],
      ),
    );
  }

  Widget _controlButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
  }) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        IconButton(
          icon: Icon(icon, color: Colors.black87),
          onPressed: onPressed,
        ),
        Text(
          label,
          style: const TextStyle(
            color: Colors.black87,
            fontSize: 10,
          ),
        ),
      ],
    );
  }
}

class GraphPainter extends CustomPainter {
  final List<DocumentNode> nodes;
  final List<DocumentEdge> edges;
  final Map<String, Offset> nodePositions;

  GraphPainter({
    required this.nodes,
    required this.edges,
    required this.nodePositions,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw edges first (so they appear behind nodes)
    for (final edge in edges) {
      final sourcePos = nodePositions[edge.sourceId];
      final targetPos = nodePositions[edge.targetId];
      
      if (sourcePos != null && targetPos != null) {
        final paint = Paint()
          ..color = _getSimilarityColor(edge.similarity).withOpacity(0.4) // Translucent
          ..strokeWidth = _getSimilarityWidth(edge.similarity)
          ..style = PaintingStyle.stroke;
        
        canvas.drawLine(sourcePos, targetPos, paint);
      }
    }
    
    // Draw nodes and labels
    for (final node in nodes) {
      final position = nodePositions[node.id];
      if (position == null) continue;
      
      // Small dot size, slightly larger if it has many connections
      final nodeRadius = 6.0 + (node.connectionCount * 0.5).clamp(0.0, 4.0);
      
      // Determine color based on whether node has connections
      final hasConnections = node.connectionCount > 0;
      final nodeColor = hasConnections ? Colors.black : Colors.grey.shade400;
      final borderColor = hasConnections ? Colors.grey.shade400 : Colors.grey.shade300;
      
      // Draw node circle (small dot)
      final circlePaint = Paint()
        ..color = nodeColor
        ..style = PaintingStyle.fill;
      
      canvas.drawCircle(position, nodeRadius, circlePaint);
      
      // Draw subtle border
      final borderPaint = Paint()
        ..color = borderColor
        ..strokeWidth = 1.5
        ..style = PaintingStyle.stroke;
      
      canvas.drawCircle(position, nodeRadius, borderPaint);
      
      // Draw file name label below the dot (centered)
      final textPainter = TextPainter(
        text: TextSpan(
          text: node.fileName,
          style: TextStyle(
            color: hasConnections ? Colors.black : Colors.grey.shade600,
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        ),
        textAlign: TextAlign.center,
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: 250);
      
      // Position label below the dot, centered
      textPainter.paint(
        canvas,
        Offset(
          position.dx - textPainter.width / 2,
          position.dy + nodeRadius + 10,
        ),
      );
    }
  }

  Color _getSimilarityColor(double similarity) {
    // All connections are 80-100% now, so use grey
    return Colors.grey.shade600;
  }

  double _getSimilarityWidth(double similarity) {
    // Thinner lines, subtle width variation
    return 1.0 + (similarity * 1.0);
  }

  @override
  bool shouldRepaint(GraphPainter oldDelegate) => false;
}
