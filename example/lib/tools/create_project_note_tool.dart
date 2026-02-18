import 'package:cactus/cactus.dart';
import '../services/rag_service.dart';

class CreateProjectNoteTool {
  static const String name = 'create_project_note';
  static const String description =
      'Create and save a note for the current project. Can generate paper summaries, '
      'project objectives, or project plans. Summaries are generated from paper content, '
      'while objectives and plans are created from user input.';

  static Map<String, dynamic> get parameters => {
        'type': 'object',
        'properties': {
          'note_type': {
            'type': 'string',
            'description': 'Type of note to create',
            'enum': ['summary', 'objective', 'plan', 'general'],
          },
          'paper_name': {
            'type': 'string',
            'description':
                'Name of paper to summarize (required for summary type)',
          },
          'content': {
            'type': 'string',
            'description':
                'User-provided content for objective/plan/general notes',
          },
        },
        'required': ['note_type'],
      };

  static Future<String> execute(
    Map<String, dynamic> params,
    ProjectService? projectService,
    RAGService? ragService,
    CactusLM? chatModel,
  ) async {
    if (projectService == null || projectService.currentProject == null) {
      return 'Error: No project is currently active. Please select or create a project first.';
    }

    if (ragService == null || chatModel == null) {
      return 'Error: Required services not initialized.';
    }

    final currentProject = projectService.currentProject!;
    final noteType = params['note_type'] as String;
    final paperName = params['paper_name'] as String?;
    final userContent = params['content'] as String?;

    try {
      String noteTitle;
      String noteContent;
      String noteCategory;

      if (noteType == 'summary') {
        // Generate paper summary
        if (paperName == null || paperName.isEmpty) {
          return 'Error: Paper name is required for summary generation.';
        }

        // Search for the paper in vector DB
        final allDocs = await ragService.getAllDocuments();
        final matchingDocs = allDocs.where((doc) {
          return doc.fileName.toLowerCase().contains(paperName.toLowerCase()) &&
              (doc.projectName == currentProject.name);
        }).toList();

        if (matchingDocs.isEmpty) {
          return 'Error: Paper "$paperName" not found in the current project. '
              'Please check the paper name or upload the paper first.';
        }

        final paper = matchingDocs.first;
        
        // Get paper content from chunks
        final searchResults = await ragService.search(
          query: paperName,
          projectName: currentProject.name,
          limit: 10,
        );

        if (searchResults.isEmpty) {
          return 'Error: Could not retrieve content for paper "${paper.fileName}".';
        }

        // Build context from chunks
        final paperContext = searchResults.map((r) => r.chunk.content).join('\n\n');

        // Generate summary using chat model
        final summaryPrompt = '''Analyze and summarize this research paper in markdown format.

Paper: ${paper.fileName}

Content:
$paperContext

Create a comprehensive summary with the following structure:

# Summary: ${paper.fileName}

## Overview
[Brief overview of what the paper is about]

## Key Objectives
[Main goals and research questions]

## Methodology
[Approach and methods used]

## Main Findings
[Key results and discoveries]

## Conclusions
[Main conclusions and implications]

## Significance
[Why this work matters]

Use clear, concise markdown formatting. Keep it informative but readable.''';

        final response = await chatModel.generateCompletion(
          messages: [
            ChatMessage(content: summaryPrompt, role: 'user'),
          ],
          params: CactusCompletionParams(
            maxTokens: 1500,
            temperature: 0.7,
          ),
        );

        if (!response.success) {
          return 'Error generating summary. Please try again.';
        }

        noteTitle = 'Summary-${paper.fileName}';
        noteContent = response.response;
        noteCategory = 'summary';

      } else if (noteType == 'objective') {
        // Create project objective note
        if (userContent == null || userContent.isEmpty) {
          return 'Error: Content is required for creating project objectives.';
        }

        final objectivePrompt = '''Format the following project objective/goals in clear markdown format.

User Input:
$userContent

Create a well-structured document with:

# Project Objective

## Goals
[List the main goals clearly]

## Background
[Context for why these objectives matter]

## Expected Outcomes
[What success looks like]

Format in clean, professional markdown.''';

        final response = await chatModel.generateCompletion(
          messages: [
            ChatMessage(content: objectivePrompt, role: 'user'),
          ],
          params: CactusCompletionParams(
            maxTokens: 1000,
            temperature: 0.7,
          ),
        );

        if (!response.success) {
          return 'Error generating objective note. Please try again.';
        }

        noteTitle = 'Project Objective';
        noteContent = response.response;
        noteCategory = 'concept';

      } else if (noteType == 'plan') {
        // Create project plan note
        if (userContent == null || userContent.isEmpty) {
          return 'Error: Content is required for creating project plans.';
        }

        final planPrompt = '''Format the following project plan in structured markdown format.

User Input:
$userContent

Create a comprehensive plan with:

# Project Plan

## Overview
[Brief description of the plan]

## Phases/Steps
[Organized breakdown of tasks and timeline]

## Resources Needed
[Key resources or requirements]

## Milestones
[Key milestones and checkpoints]

## Success Criteria
[How to measure completion]

Format in clean, actionable markdown.''';

        final response = await chatModel.generateCompletion(
          messages: [
            ChatMessage(content: planPrompt, role: 'user'),
          ],
          params: CactusCompletionParams(
            maxTokens: 1200,
            temperature: 0.7,
          ),
        );

        if (!response.success) {
          return 'Error generating plan note. Please try again.';
        }

        noteTitle = 'Project Plan';
        noteContent = response.response;
        noteCategory = 'writeup';

      } else {
        // General note
        if (userContent == null || userContent.isEmpty) {
          return 'Error: Content is required for creating general notes.';
        }

        noteTitle = 'Note-${DateTime.now().millisecondsSinceEpoch}';
        noteContent = userContent;
        noteCategory = 'general';
      }

      // Save the note to the project
      print('>>> Saving note to database...');
      await projectService.createNote(
        projectId: currentProject.id,
        title: noteTitle,
        content: noteContent,
        noteType: noteCategory,
      );
      print('>>> Note saved to database successfully');

      // Also save to file system
      print('>>> Saving note to file system...');
      await projectService.storageService.saveNote(
        projectName: currentProject.name,
        noteTitle: noteTitle,
        content: noteContent,
      );
      print('>>> Note saved to file system successfully');

      // Return a concise confirmation message
      print('>>> Building confirmation message...');
      final buffer = StringBuffer();
      buffer.writeln('Successfully created and saved note to ${currentProject.name}');
      buffer.writeln();
      buffer.writeln('Note Details:');
      buffer.writeln('  Title: $noteTitle');
      buffer.writeln('  Type: $noteCategory');
      buffer.writeln('  Location: Notes section');
      buffer.writeln();
      buffer.writeln('You can view the full note in the Notes tab of this project.');
      
      final confirmationMessage = buffer.toString();
      print('>>> Returning confirmation message:');
      print(confirmationMessage);
      print('>>> TOOL EXECUTION COMPLETE');
      
      return confirmationMessage;

    } catch (e) {
      return 'Error creating note: $e';
    }
  }
}
