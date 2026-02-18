import 'package:objectbox/objectbox.dart';

@Entity()
class ProjectEntity {
  @Id()
  int objectId = 0;

  @Unique()
  String id;

  String name;
  String? description;

  int createdAt;
  int updatedAt;

  ProjectEntity({
    required this.id,
    required this.name,
    this.description,
    required this.createdAt,
    required this.updatedAt,
  });
}
