import 'package:flutter/material.dart';
import 'package:cactus/memory/conversation_store.dart';

class ConversationDrawer extends StatefulWidget {
  final ConversationStore conversationStore;
  final String currentConversationId;
  final Function(String) onSelectConversation;
  final VoidCallback onNewConversation;

  const ConversationDrawer({
    super.key,
    required this.conversationStore,
    required this.currentConversationId,
    required this.onSelectConversation,
    required this.onNewConversation,
  });

  @override
  State<ConversationDrawer> createState() => _ConversationDrawerState();
}

class _ConversationDrawerState extends State<ConversationDrawer> {
  late List<Conversation> _conversations;

  @override
  void initState() {
    super.initState();
    _loadConversations();
  }

  void _loadConversations() {
    setState(() {
      _conversations = widget.conversationStore.getAllConversations();
    });
  }

  Future<void> _deleteConversation(String conversationId) async {
    await widget.conversationStore.deleteConversation(conversationId);
    _loadConversations();

    if (conversationId == widget.currentConversationId) {
      if (_conversations.isNotEmpty) {
        widget.onSelectConversation(_conversations.first.id);
      } else {
        widget.onNewConversation();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Drawer(
      width: MediaQuery.of(context).size.width,
      child: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Conversations',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),
            const Divider(),
            Expanded(
              child: _conversations.isEmpty
                  ? const Center(
                      child: Text(
                        'No conversations yet',
                        style: TextStyle(color: Colors.grey),
                      ),
                    )
                  : ListView.builder(
                      itemCount: _conversations.length,
                      itemBuilder: (context, index) {
                        final convo = _conversations[index];
                        final isSelected = convo.id == widget.currentConversationId;

                        return GestureDetector(
                          onLongPress: () => _showDeleteDialog(convo),
                          child: ListTile(
                            selected: isSelected,
                            selectedTileColor: Colors.blue.withOpacity(0.1),
                            leading: Icon(
                              convo.isVoiceMode ? Icons.mic : Icons.chat_bubble_outline,
                              color: isSelected ? Colors.blue : Colors.grey,
                            ),
                            title: Text(
                              convo.title,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                              ),
                            ),
                            subtitle: Text(
                              _formatDate(convo.updatedAt),
                              style: const TextStyle(fontSize: 12),
                            ),
                            onTap: () {
                              widget.onSelectConversation(convo.id);
                              Navigator.pop(context);
                            },
                          ),
                        );
                      },
                    ),
            ),
            const Divider(),
            Padding(
              padding: const EdgeInsets.all(16),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () {
                    widget.onNewConversation();
                    Navigator.pop(context);
                  },
                  icon: const Icon(Icons.add),
                  label: const Text('New Conversation'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showDeleteDialog(Conversation convo) {
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
              _deleteConversation(convo.id);
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final diff = now.difference(date);

    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inHours < 1) return '${diff.inMinutes}m ago';
    if (diff.inDays < 1) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${date.day}/${date.month}/${date.year}';
  }
}
