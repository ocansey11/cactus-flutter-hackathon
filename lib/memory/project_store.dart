import 'package:uuid/uuid.dart';
import 'objectbox_manager.dart';
import 'entities/project_entity.dart';
import '../objectbox.g.dart';

class Project {
  final String id;
  final String name;
  final String? description;
  final DateTime createdAt;
  final DateTime updatedAt;

  Project({
    required this.id,
    required this.name,
    this.description,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Project.fromEntity(ProjectEntity entity) {
    return Project(
      id: entity.id,
      name: entity.name,
      description: entity.description,
      createdAt: DateTime.fromMillisecondsSinceEpoch(entity.createdAt),
      updatedAt: DateTime.fromMillisecondsSinceEpoch(entity.updatedAt),
    );
  }
}

class ProjectStore {
  static const _uuid = Uuid();

  Project createProject({required String name, String? description}) {
    final now = DateTime.now();
    final entity = ProjectEntity(
      id: _uuid.v4(),
      name: name,
      description: description,
      createdAt: now.millisecondsSinceEpoch,
      updatedAt: now.millisecondsSinceEpoch,
    );
    ObjectBoxManager.projects.put(entity);
    return Project.fromEntity(entity);
  }

  List<Project> getAllProjects() {
    final entities = ObjectBoxManager.projects.getAll();
    final projects = entities.map((e) => Project.fromEntity(e)).toList();
    projects.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    return projects;
  }

  Project? getProject(String projectId) {
    final query = ObjectBoxManager.projects
        .query(ProjectEntity_.id.equals(projectId))
        .build();
    final entity = query.findFirst();
    query.close();
    return entity != null ? Project.fromEntity(entity) : null;
  }

  void updateProject(String projectId, {String? name, String? description}) {
    final query = ObjectBoxManager.projects
        .query(ProjectEntity_.id.equals(projectId))
        .build();
    final entity = query.findFirst();
    query.close();

    if (entity == null) return;
    if (name != null) entity.name = name;
    if (description != null) entity.description = description;
    entity.updatedAt = DateTime.now().millisecondsSinceEpoch;
    ObjectBoxManager.projects.put(entity);
  }

  void deleteProject(String projectId) {
    final query = ObjectBoxManager.projects
        .query(ProjectEntity_.id.equals(projectId))
        .build();
    final entity = query.findFirst();
    query.close();
    if (entity != null) ObjectBoxManager.projects.remove(entity.objectId);
  }
}
