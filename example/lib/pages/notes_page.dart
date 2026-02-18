import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart' show Note;
import '../services/project_service.dart';

class NotesPage extends StatefulWidget {
  final ProjectService projectService;

  const NotesPage({
    super.key,
    required this.projectService,
  });

  @override
  State<NotesPage> createState() => _NotesPageState();
}

class _NotesPageState extends State<NotesPage> {
  List<Note> _notes = [];

  @override
  void initState() {
    super.initState();
    _loadNotes();
  }

  void _loadNotes() {
    final project = widget.projectService.currentProject;
    if (project == null) return;
    setState(() {
      _notes = widget.projectService.getNotes(project.id);
    });
  }

  String _formatDate(DateTime date) {
    final diff = DateTime.now().difference(date);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inHours < 1) return '${diff.inMinutes}m ago';
    if (diff.inDays < 1) return '${diff.inHours}h ago';
    if (diff.inDays == 1) return 'Yesterday';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${date.day}/${date.month}/${date.year}';
  }

  Color _colorForType(String type) {
    switch (type) {
      case 'summary':
        return Colors.blue;
      case 'objective':
        return Colors.purple;
      case 'plan':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  IconData _iconForType(String type) {
    switch (type) {
      case 'summary':
        return Icons.summarize_outlined;
      case 'objective':
        return Icons.flag_outlined;
      case 'plan':
        return Icons.checklist_outlined;
      default:
        return Icons.note_outlined;
    }
  }

  @override
  Widget build(BuildContext context) {
    final project = widget.projectService.currentProject;

    return Scaffold(
      appBar: AppBar(
        title: Text('${project?.name ?? 'Project'} — Notes'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: _createNote,
            tooltip: 'Create Note',
          ),
        ],
      ),
      body: _notes.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.note_add_outlined,
                      size: 64, color: Colors.grey[400]),
                  const SizedBox(height: 16),
                  Text('No notes yet',
                      style:
                          TextStyle(fontSize: 18, color: Colors.grey[600])),
                  const SizedBox(height: 8),
                  Text('Ask the model to create notes, or tap +',
                      style:
                          TextStyle(fontSize: 14, color: Colors.grey[500])),
                  const SizedBox(height: 24),
                  ElevatedButton.icon(
                    onPressed: _createNote,
                    icon: const Icon(Icons.add),
                    label: const Text('Create Note'),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(8),
              itemCount: _notes.length,
              itemBuilder: (context, index) {
                final note = _notes[index];
                return Card(
                  margin: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 4),
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor: _colorForType(note.noteType),
                      child: Icon(_iconForType(note.noteType),
                          color: Colors.white, size: 20),
                    ),
                    title: Text(note.title,
                        style: const TextStyle(
                            fontWeight: FontWeight.w500)),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const SizedBox(height: 4),
                        Text(
                          note.content.length > 100
                              ? '${note.content.substring(0, 100)}...'
                              : note.content,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                              fontSize: 13, color: Colors.grey[700]),
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Chip(
                              label: Text(note.noteType.toUpperCase(),
                                  style:
                                      const TextStyle(fontSize: 10)),
                              backgroundColor: _colorForType(note.noteType)
                                  .withOpacity(0.1),
                              labelPadding:
                                  const EdgeInsets.symmetric(horizontal: 4),
                              visualDensity: VisualDensity.compact,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              _formatDate(note.createdAt),
                              style: TextStyle(
                                  fontSize: 12, color: Colors.grey[500]),
                            ),
                          ],
                        ),
                      ],
                    ),
                    trailing: IconButton(
                      icon: const Icon(Icons.more_vert),
                      onPressed: () => _showOptions(note),
                    ),
                    onTap: () => _viewNote(note),
                  ),
                );
              },
            ),
      floatingActionButton: _notes.isEmpty
          ? null
          : FloatingActionButton(
              onPressed: _createNote,
              child: const Icon(Icons.add),
            ),
    );
  }

  void _createNote() {
    showDialog(
      context: context,
      builder: (context) => _NoteEditorDialog(
        projectService: widget.projectService,
        onSaved: _loadNotes,
      ),
    );
  }

  void _viewNote(Note note) {
    showDialog(
      context: context,
      builder: (context) => _NoteEditorDialog(
        projectService: widget.projectService,
        note: note,
        onSaved: _loadNotes,
      ),
    );
  }

  void _showOptions(Note note) {
    showModalBottomSheet(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.edit_outlined),
              title: const Text('Edit'),
              onTap: () {
                Navigator.pop(context);
                _viewNote(note);
              },
            ),
            ListTile(
              leading:
                  const Icon(Icons.delete_outline, color: Colors.red),
              title: const Text('Delete',
                  style: TextStyle(color: Colors.red)),
              onTap: () {
                Navigator.pop(context);
                _confirmDelete(note);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _confirmDelete(Note note) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Note'),
        content: Text('Delete "${note.title}"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              widget.projectService.deleteNote(note.id);
              _loadNotes();
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Note deleted')),
                );
              }
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}

class _NoteEditorDialog extends StatefulWidget {
  final ProjectService projectService;
  final Note? note;
  final VoidCallback onSaved;

  const _NoteEditorDialog({
    required this.projectService,
    this.note,
    required this.onSaved,
  });

  @override
  State<_NoteEditorDialog> createState() => _NoteEditorDialogState();
}

class _NoteEditorDialogState extends State<_NoteEditorDialog> {
  late TextEditingController _titleController;
  late TextEditingController _contentController;
  String _selectedType = 'general';

  @override
  void initState() {
    super.initState();
    _titleController =
        TextEditingController(text: widget.note?.title ?? '');
    _contentController =
        TextEditingController(text: widget.note?.content ?? '');
    _selectedType = widget.note?.noteType ?? 'general';
  }

  @override
  void dispose() {
    _titleController.dispose();
    _contentController.dispose();
    super.dispose();
  }

  void _save() {
    if (_titleController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a title')),
      );
      return;
    }

    final project = widget.projectService.currentProject;
    if (project == null) return;

    if (widget.note == null) {
      widget.projectService.createNote(
        projectId: project.id,
        conversationId: 'manual',
        title: _titleController.text.trim(),
        content: _contentController.text,
        noteType: _selectedType,
      );
    } else {
      widget.projectService.updateNote(
        noteId: widget.note!.id,
        title: _titleController.text.trim(),
        content: _contentController.text,
        noteType: _selectedType,
      );
    }

    Navigator.pop(context);
    widget.onSaved();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
            widget.note == null ? 'Note created' : 'Note updated'),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.9,
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              widget.note == null ? 'Create Note' : 'Edit Note',
              style: const TextStyle(
                  fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _titleController,
              decoration: const InputDecoration(
                labelText: 'Title',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              value: _selectedType,
              decoration: const InputDecoration(
                labelText: 'Type',
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem(value: 'general', child: Text('General')),
                DropdownMenuItem(value: 'summary', child: Text('Summary')),
                DropdownMenuItem(
                    value: 'objective', child: Text('Objective')),
                DropdownMenuItem(value: 'plan', child: Text('Plan')),
              ],
              onChanged: (value) {
                if (value != null) setState(() => _selectedType = value);
              },
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _contentController,
              decoration: const InputDecoration(
                labelText: 'Content',
                border: OutlineInputBorder(),
                alignLabelWithHint: true,
              ),
              maxLines: 10,
              minLines: 5,
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Cancel'),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _save,
                  child: const Text('Save'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
