import 'package:flutter/material.dart';
import '../services/permission_service.dart';
import '../services/voice_service.dart';
import '../services/voice_conversation_service.dart';
import '../services/conversation_service.dart';
import 'package:cactus/memory/conversation_store.dart'; 

enum VoiceState {
  idle,
  listening,
  processing,
  speaking,
}

class VoiceChatPage extends StatefulWidget {
  final ConversationService conversationService;
  final ConversationStore conversationStore;      
  final String currentConversationId;             
  final VoidCallback? onSwitchMode;
  final VoidCallback? onOpenDrawer;
  
  const VoiceChatPage({
    super.key,
    required this.conversationService,
    required this.conversationStore,              
    required this.currentConversationId,          
    this.onSwitchMode,
    this.onOpenDrawer,
  });

  @override
  State<VoiceChatPage> createState() => _VoiceChatPageState();
}

class _VoiceChatPageState extends State<VoiceChatPage> {
  final VoiceService _voiceService = VoiceService();
  late VoiceConversationService _voiceConversation;
  
  VoiceState _state = VoiceState.idle;
  String _statusText = "Initializing...";
  bool _hasPermission = false;
  bool _isInitialized = false;
  
  List<Message> _messages = [];  

  @override
  void initState() {
    super.initState();
    _loadMessages();  
    _initializeVoice();
  }

  @override
  void dispose() {
    _voiceService.dispose();
    super.dispose();
  }

  void _loadMessages() {
    setState(() {
      _messages = widget.conversationStore.getMessages(widget.currentConversationId);
    });
  }

  Future<void> _initializeVoice() async {
    final hasPermission = await PermissionService.checkAndRequestMicrophone();
    setState(() => _hasPermission = hasPermission);
    
    if (!hasPermission) {
      setState(() => _statusText = "Microphone permission required");
      _showPermissionDialog();
      return;
    }

    try {
      setState(() => _statusText = "Loading voice models...");
      
      await _voiceService.initialize(model: 'whisper-base');
      
      _voiceConversation = VoiceConversationService(
        voiceService: _voiceService,
        conversationService: widget.conversationService,
      );
      
      setState(() {
        _isInitialized = true;
        _statusText = "Hold button to speak";
      });
      
      await _voiceService.speak("Hello, I'm Cactus. How can I help you?");
    } catch (e) {
      setState(() => _statusText = "Initialization failed: $e");
    }
  }

  void _showPermissionDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Microphone Permission Required'),
        content: const Text(
          'Please enable microphone access in settings to use voice features.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.menu),
          onPressed: widget.onOpenDrawer,
        ),
        title: const Text('Voice Chat'),
        actions: [
          IconButton(
            icon: const Icon(Icons.chat),
            onPressed: widget.onSwitchMode,
            tooltip: 'Switch to Text Chat',
          ),
        ],
      ),
      body: Column(
        children: [
          // ← ADD MESSAGE HISTORY
          Expanded(
            child: _messages.isEmpty
                ? const Center(
                    child: Text(
                      'Hold button and speak to begin',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.grey,
                      ),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      final msg = _messages[index];
                      return _buildMessageBubble(msg);
                    },
                  ),
          ),
          
          // Voice controls at bottom
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 10,
                  offset: const Offset(0, -5),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildOrb(),
                const SizedBox(height: 20),
                Text(
                  _statusText,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                  textAlign: TextAlign.center,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 20),
                _buildMicButton(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageBubble(Message message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        decoration: BoxDecoration(
          color: message.isUser ? Colors.blue : Colors.grey[200],
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          message.text,
          style: TextStyle(
            color: message.isUser ? Colors.white : Colors.black,
            fontSize: 15,
          ),
        ),
      ),
    );
  }

  Widget _buildOrb() {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: 150,
      height: 150,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: _getOrbColor(),
        boxShadow: _state == VoiceState.listening
            ? [
                BoxShadow(
                  color: Colors.red.withOpacity(0.5),
                  blurRadius: 40,
                  spreadRadius: 10,
                ),
              ]
            : [
                BoxShadow(
                  color: Colors.blue.withOpacity(0.3),
                  blurRadius: 20,
                  spreadRadius: 5,
                ),
              ],
      ),
      child: Center(
        child: _getOrbIcon(),
      ),
    );
  }

  Color _getOrbColor() {
    switch (_state) {
      case VoiceState.idle:
        return Colors.blue;
      case VoiceState.listening:
        return Colors.red;
      case VoiceState.processing:
        return Colors.orange;
      case VoiceState.speaking:
        return Colors.green;
    }
  }

  Widget _getOrbIcon() {
    switch (_state) {
      case VoiceState.idle:
        return const Icon(
          Icons.mic_none,
          size: 60,
          color: Colors.white,
        );
      case VoiceState.listening:
        return const Icon(
          Icons.mic,
          size: 60,
          color: Colors.white,
        );
      case VoiceState.processing:
        return const CircularProgressIndicator(
          color: Colors.white,
          strokeWidth: 5,
        );
      case VoiceState.speaking:
        return const Icon(
          Icons.volume_up,
          size: 60,
          color: Colors.white,
        );
    }
  }

  Widget _buildMicButton() {
    final isEnabled = _hasPermission && _isInitialized && _state != VoiceState.processing;
    
    return GestureDetector(
      onTapDown: isEnabled ? (_) => _startListening() : null,
      onTapUp: isEnabled ? (_) => _stopListening() : null,
      onTapCancel: isEnabled ? () => _stopListening() : null,
      child: Container(
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isEnabled ? Colors.blue : Colors.grey,
        ),
        child: Icon(
          _state == VoiceState.listening ? Icons.mic : Icons.mic_none,
          color: Colors.white,
          size: 40,
        ),
      ),
    );
  }

  Future<void> _startListening() async {
    if (!_hasPermission || !_isInitialized) {
      _showPermissionDialog();
      return;
    }

    try {
      setState(() {
        _state = VoiceState.listening;
        _statusText = "Listening...";
      });
      
      await _voiceService.startRecording();
    } catch (e) {
      setState(() {
        _state = VoiceState.idle;
        _statusText = "Error: $e";
      });
    }
  }

  Future<void> _stopListening() async {
    if (_state != VoiceState.listening) return;

    setState(() {
      _state = VoiceState.processing;
      _statusText = "Processing...";
    });

    try {
      final transcription = await _voiceService.stopRecording();
      
      if (transcription.isEmpty) {
        setState(() {
          _state = VoiceState.idle;
          _statusText = "No speech detected. Hold button to speak";
        });
        return;
      }

      // ← STORE USER MESSAGE
      await widget.conversationStore.addMessage(
        conversationId: widget.currentConversationId,
        text: transcription,
        isUser: true,
      );
      _loadMessages();  // Refresh UI

      setState(() => _statusText = "You said: $transcription");
      
      final response = await _voiceConversation.processVoiceInput(transcription);
      
      // ← STORE ASSISTANT MESSAGE
      await widget.conversationStore.addMessage(
        conversationId: widget.currentConversationId,
        text: response,
        isUser: false,
      );
      _loadMessages();  // Refresh UI
      
      setState(() {
        _state = VoiceState.speaking;
        _statusText = response;
      });
      
      await _voiceService.speak(response);
      
      await Future.delayed(const Duration(seconds: 1));
      
      if (mounted) {
        setState(() {
          _state = VoiceState.idle;
          _statusText = "Hold button to speak";
        });
      }
    } catch (e) {
      setState(() {
        _state = VoiceState.idle;
        _statusText = "Error: $e";
      });
    }
  }
}