import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';
import '../services/project_service.dart';

class AppDrawer extends StatefulWidget {
  final ProjectService projectService;
  final String? currentConversationId;
  final Function(Project project, Conversation conversation) onSelectConversation;
  final Future<void> Function(Project project) onNewConversation;

  const AppDrawer({
    super.key,
    required this.projectService,
    required this.currentConversationId,
    required this.onSelectConversation,
    required this.onNewConversation,
  });

  @override
  State<AppDrawer> createState() => _AppDrawerState();
}

class _AppDrawerState extends State<AppDrawer> {
  List<Project> _projects = [];
  // Track which projects are expanded in the drawer
  final Set<String> _expanded = {};

  @override
  void initState() {
    super.initState();
    _load();
    // Auto-expand the current project
    final current = widget.projectService.currentProject;
    if (current != null) _expanded.add(current.id);
  }

  void _load() {
    setState(() {
      _projects = widget.projectService.getAllProjects();
    });
  }

  String _formatDate(DateTime date) {
    final diff = DateTime.now().difference(date);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inHours < 1) return '${diff.inMinutes}m ago';
    if (diff.inDays < 1) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${date.day}/${date.month}/${date.year}';
  }

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: SafeArea(
        child: Column(
          children: [
            // Header
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
              child: Row(
                children: [
                  const Icon(Icons.memory, size: 28),
                  const SizedBox(width: 10),
                  const Expanded(
                    child: Text(
                      'JarvisOS',
                      style: TextStyle(
                          fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),
            const Divider(height: 1),

            // Projects + conversations
            Expanded(
              child: _projects.isEmpty
                  ? const Center(
                      child: Text('No projects yet',
                          style: TextStyle(color: Colors.grey)),
                    )
                  : ListView.builder(
                      itemCount: _projects.length,
                      itemBuilder: (context, i) =>
                          _buildProjectTile(_projects[i]),
                    ),
            ),

            const Divider(height: 1),

            // New project button
            Padding(
              padding: const EdgeInsets.all(12),
              child: SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                    _createProject(context);
                  },
                  icon: const Icon(Icons.create_new_folder_outlined),
                  label: const Text('New Project'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProjectTile(Project project) {
    final isCurrentProject =
        widget.projectService.currentProject?.id == project.id;
    final isExpanded = _expanded.contains(project.id);
    final conversations =
        widget.projectService.getConversations(projectId: project.id);

    return Column(
      children: [
        // Project header row
        InkWell(
          onTap: () {
            setState(() {
              if (isExpanded) {
                _expanded.remove(project.id);
              } else {
                _expanded.add(project.id);
              }
            });
          },
          child: Container(
            color: isCurrentProject
                ? Theme.of(context)
                    .colorScheme
                    .primaryContainer
                    .withOpacity(0.4)
                : null,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            child: Row(
              children: [
                Icon(
                  Icons.folder_outlined,
                  size: 20,
                  color: isCurrentProject
                      ? Theme.of(context).colorScheme.primary
                      : Colors.grey[600],
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    project.name,
                    style: TextStyle(
                      fontWeight: isCurrentProject
                          ? FontWeight.bold
                          : FontWeight.normal,
                      fontSize: 15,
                    ),
                  ),
                ),
                // New conversation button
                IconButton(
                  icon: const Icon(Icons.add, size: 18),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  tooltip: 'New conversation',
                  onPressed: () async {
                    widget.projectService.setCurrentProject(project);
                    await widget.onNewConversation(project);
                    if (context.mounted) Navigator.pop(context);
                  },
                ),
                const SizedBox(width: 4),
                Icon(
                  isExpanded
                      ? Icons.expand_less
                      : Icons.expand_more,
                  size: 18,
                  color: Colors.grey,
                ),
              ],
            ),
          ),
        ),

        // Conversations list
        if (isExpanded)
          conversations.isEmpty
              ? Padding(
                  padding: const EdgeInsets.only(left: 48, bottom: 8),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'No conversations',
                      style: TextStyle(
                          fontSize: 12, color: Colors.grey[500]),
                    ),
                  ),
                )
              : Column(
                  children: conversations.map((convo) {
                    final isSelected =
                        convo.id == widget.currentConversationId;
                    return InkWell(
                      onLongPress: () => _confirmDeleteConvo(convo),
                      child: Container(
                        color: isSelected
                            ? Theme.of(context)
                                .colorScheme
                                .primary
                                .withOpacity(0.1)
                            : null,
                        padding: const EdgeInsets.only(
                            left: 48, right: 16, top: 8, bottom: 8),
                        child: Row(
                          children: [
                            Icon(
                              Icons.chat_bubble_outline,
                              size: 16,
                              color: isSelected
                                  ? Theme.of(context).colorScheme.primary
                                  : Colors.grey,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Column(
                                crossAxisAlignment:
                                    CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    convo.title,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: TextStyle(
                                      fontSize: 13,
                                      fontWeight: isSelected
                                          ? FontWeight.w600
                                          : FontWeight.normal,
                                    ),
                                  ),
                                  Text(
                                    _formatDate(convo.updatedAt),
                                    style: TextStyle(
                                        fontSize: 11,
                                        color: Colors.grey[500]),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                      onTap: () {
                        widget.projectService.setCurrentProject(project);
                        widget.onSelectConversation(project, convo);
                        Navigator.pop(context);
                      },
                    );
                  }).toList(),
                ),
      ],
    );
  }

  void _confirmDeleteConvo(Conversation convo) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Conversation'),
        content: Text('Delete "${convo.title}"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              widget.projectService.deleteConversation(convo.id);
              _load();
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  void _createProject(BuildContext context) {
    final nameController = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('New Project'),
        content: TextField(
          controller: nameController,
          decoration: const InputDecoration(
            labelText: 'Project name',
            border: OutlineInputBorder(),
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              if (nameController.text.trim().isNotEmpty) {
                widget.projectService
                    .createProject(name: nameController.text.trim());
                Navigator.pop(context);
                _load();
              }
            },
            child: const Text('Create'),
          ),
        ],
      ),
    );
  }
}
