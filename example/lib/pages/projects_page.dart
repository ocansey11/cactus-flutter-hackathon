import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import '../services/conversation_service.dart';
import 'rag_chat.dart';
import 'document_graph_page.dart';
import '../tools/document_similarity_tool.dart';

class ProjectsPage extends StatefulWidget {
  final ProjectService projectService;
  final ConversationService conversationService;

  const ProjectsPage({
    super.key,
    required this.projectService,
    required this.conversationService,
  });

  @override
  State<ProjectsPage> createState() => _ProjectsPageState();
}

class _ProjectsPageState extends State<ProjectsPage> {
  List<ResearchProject> _projects = [];

  @override
  void initState() {
    super.initState();
    _loadProjects();
  }

  void _loadProjects() {
    setState(() {
      _projects = widget.projectService.getAllProjects();
    });
  }

  Future<void> _createProject() async {
    final nameController = TextEditingController();
    final descController = TextEditingController();

    final result = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Create New Project'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameController,
              decoration: const InputDecoration(
                labelText: 'Project Name',
                hintText: 'e.g., AI Ethics Research',
              ),
              autofocus: true,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: descController,
              decoration: const InputDecoration(
                labelText: 'Description (optional)',
                hintText: 'Brief description of the project',
              ),
              maxLines: 3,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Create'),
          ),
        ],
      ),
    );

    if (result == true && nameController.text.isNotEmpty) {
      await widget.projectService.createProject(
        name: nameController.text.trim(),
        description: descController.text.trim().isEmpty
            ? null
            : descController.text.trim(),
      );
      _loadProjects();
    }
  }

  Future<void> _selectProject(ResearchProject project) async {
    widget.projectService.setCurrentProject(project);
    
    if (mounted) {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => RAGChatPage(
            conversationService: widget.conversationService,
            projectService: widget.projectService,
          ),
        ),
      );
      
      setState(() {});
    }
  }

  Future<void> _deleteProject(ResearchProject project) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Project'),
        content: Text(
          'Are you sure you want to delete "${project.name}"? This will remove all associated files and notes.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            style: FilledButton.styleFrom(
              backgroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await widget.projectService.deleteProject(project.id);
      _loadProjects();
    }
  }

  Future<void> _openGraphView() async {
    try {
      final result = await DocumentSimilarityTool.execute(
        {'threshold': 0.8},
        widget.conversationService.ragService,
      );

      if (DocumentSimilarityTool.lastComputedGraph != null) {
        final graph = DocumentSimilarityTool.lastComputedGraph!;

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => DocumentGraphPage(
                graphData: graph,
              ),
            ),
          );
        }
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No documents to compare')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  Future<void> _clearVectorDatabase() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Vector Database?'),
        content: const Text(
          'This will permanently delete all documents and embeddings from the vector database. '
          'Project folders and files will not be affected.\n\n'
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            style: FilledButton.styleFrom(
              backgroundColor: Colors.red,
            ),
            child: const Text('Clear Database'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      try {
        await widget.conversationService.ragService?.clearDatabase();
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Vector database cleared successfully'),
              backgroundColor: Colors.green,
            ),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error clearing database: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final currentProject = widget.projectService.currentProject;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Research Projects'),
        actions: [
          IconButton(
            icon: const Icon(Icons.hub),
            onPressed: _openGraphView,
            tooltip: 'Document Similarity Graph',
          ),
          PopupMenuButton<String>(
            onSelected: (value) {
              if (value == 'clear_db') {
                _clearVectorDatabase();
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'clear_db',
                child: Row(
                  children: [
                    Icon(Icons.delete_sweep, color: Colors.red),
                    SizedBox(width: 8),
                    Text('Clear Vector DB'),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: _projects.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.folder_open,
                    size: 80,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'No projects yet',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          color: Colors.grey[600],
                        ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Create a project to organize your research',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: Colors.grey[500],
                        ),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _projects.length,
              itemBuilder: (context, index) {
                final project = _projects[index];
                final isSelected = currentProject?.id == project.id;
                final stats = widget.projectService.getProjectStats(project.id);

                return Card(
                  elevation: isSelected ? 4 : 1,
                  color: isSelected
                      ? Theme.of(context).colorScheme.primaryContainer
                      : null,
                  margin: const EdgeInsets.only(bottom: 12),
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor: isSelected
                          ? Theme.of(context).colorScheme.primary
                          : Colors.blue,
                      child: Icon(
                        isSelected ? Icons.check : Icons.folder,
                        color: Colors.white,
                      ),
                    ),
                    title: Text(
                      project.name,
                      style: TextStyle(
                        fontWeight:
                            isSelected ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (project.description != null) ...[
                          const SizedBox(height: 4),
                          Text(project.description!),
                        ],
                        const SizedBox(height: 8),
                        Wrap(
                          spacing: 12,
                          children: [
                            Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(Icons.description,
                                    size: 16, color: Colors.grey),
                                const SizedBox(width: 4),
                                Text('${stats.documentCount} papers'),
                              ],
                            ),
                            Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(Icons.note,
                                    size: 16, color: Colors.grey),
                                const SizedBox(width: 4),
                                Text('${stats.noteCount} notes'),
                              ],
                            ),
                          ],
                        ),
                      ],
                    ),
                    trailing: PopupMenuButton(
                      itemBuilder: (context) => [
                        if (!isSelected)
                          const PopupMenuItem(
                            value: 'select',
                            child: Row(
                              children: [
                                Icon(Icons.check_circle_outline),
                                SizedBox(width: 8),
                                Text('Select'),
                              ],
                            ),
                          ),
                        const PopupMenuItem(
                          value: 'delete',
                          child: Row(
                            children: [
                              Icon(Icons.delete_outline, color: Colors.red),
                              SizedBox(width: 8),
                              Text('Delete',
                                  style: TextStyle(color: Colors.red)),
                            ],
                          ),
                        ),
                      ],
                      onSelected: (value) {
                        switch (value) {
                          case 'select':
                            _selectProject(project);
                            break;
                          case 'delete':
                            _deleteProject(project);
                            break;
                        }
                      },
                    ),
                    onTap: () => _selectProject(project),
                  ),
                );
              },
            ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _createProject,
        icon: const Icon(Icons.add),
        label: const Text('New Project'),
      ),
    );
  }
}
