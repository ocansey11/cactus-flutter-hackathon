import 'package:permission_handler/permission_handler.dart';

class PermissionService {
  static Future<bool> requestMicrophone() async {
    final status = await Permission.microphone.request();
    return status.isGranted;
  }
  
  static Future<bool> checkMicrophone() async {
    final status = await Permission.microphone.status;
    return status.isGranted;
  }
  
  static Future<bool> checkAndRequestMicrophone() async {
    final hasPermission = await checkMicrophone();
    if (hasPermission) return true;
    return await requestMicrophone();
  }
}
