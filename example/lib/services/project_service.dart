import 'package:cactus/cactus.dart';

class ProjectService {
  final ProjectStore _projectStore;
  final ConversationStore _conversationStore;
  final DocumentMetadataStore _metadataStore;
  final NoteStore _noteStore;

  Project? _currentProject;

  ProjectService({
    required ProjectStore projectStore,
    required ConversationStore conversationStore,
    required DocumentMetadataStore metadataStore,
    required NoteStore noteStore,
  })  : _projectStore = projectStore,
        _conversationStore = conversationStore,
        _metadataStore = metadataStore,
        _noteStore = noteStore;

  Project? get currentProject => _currentProject;

  void setCurrentProject(Project project) {
    _currentProject = project;
  }

  void clearCurrentProject() {
    _currentProject = null;
  }

  Project createProject({required String name, String? description}) {
    return _projectStore.createProject(name: name, description: description);
  }

  List<Project> getAllProjects() => _projectStore.getAllProjects();

  void updateProject(String projectId, {String? name, String? description}) {
    _projectStore.updateProject(projectId, name: name, description: description);
  }

  void deleteProject(String projectId) {
    _metadataStore.deleteAllForProject(projectId);
    _noteStore.deleteAllForProject(projectId);

    final convos = _conversationStore.getAllConversations(projectId: projectId);
    for (final convo in convos) {
      _conversationStore.deleteConversation(convo.id);
    }

    _projectStore.deleteProject(projectId);

    if (_currentProject?.id == projectId) {
      _currentProject = null;
    }
  }

  Future<Conversation> createConversation({
    String? title,
    String? projectId,
    bool isVoiceMode = false,
  }) {
    return _conversationStore.createConversation(
      title: title,
      projectId: projectId,
      isVoiceMode: isVoiceMode,
    );
  }

  List<Conversation> getConversations({String? projectId}) {
    return _conversationStore.getAllConversations(projectId: projectId);
  }

  void deleteConversation(String conversationId) {
    _conversationStore.deleteConversation(conversationId);
  }

  Future<Message> addMessage({
    required String conversationId,
    required String text,
    required bool isUser,
    Map<String, dynamic>? metadata,
  }) {
    return _conversationStore.addMessage(
      conversationId: conversationId,
      text: text,
      isUser: isUser,
      metadata: metadata,
    );
  }

  List<Message> getMessages(String conversationId) {
    return _conversationStore.getMessages(conversationId);
  }

  List<ChatMessage> toChatMessages(String conversationId) {
    return _conversationStore.toChatMessages(conversationId);
  }

  String registerDocument({
    required String projectId,
    required String fileName,
    required String filePath,
    required int fileSize,
  }) {
    return _metadataStore.registerDocument(
      projectId: projectId,
      fileName: fileName,
      filePath: filePath,
      fileSize: fileSize,
    );
  }

  List<DocumentMetadata> getDocuments(String projectId) {
    return _metadataStore.getDocumentsForProject(projectId);
  }

  void deleteDocument(String docId) {
    _metadataStore.deleteDocument(docId);
  }

  Note createNote({
    required String projectId,
    required String conversationId,
    required String title,
    required String content,
    required String noteType,
    List<String> referencedPapers = const [],
  }) {
    return _noteStore.createNote(
      projectId: projectId,
      conversationId: conversationId,
      title: title,
      content: content,
      noteType: noteType,
      referencedPapers: referencedPapers,
    );
  }

  List<Note> getNotes(String projectId) {
    return _noteStore.getNotesForProject(projectId);
  }

  Note? getNoteForConversation(String conversationId) {
    return _noteStore.getNoteForConversation(conversationId);
  }

  void updateNote({
    required String noteId,
    String? title,
    String? content,
    String? noteType,
    List<String>? referencedPapers,
  }) {
    _noteStore.updateNote(
      noteId: noteId,
      title: title,
      content: content,
      noteType: noteType,
      referencedPapers: referencedPapers,
    );
  }

  void deleteNote(String noteId) {
    _noteStore.deleteNote(noteId);
  }
}
