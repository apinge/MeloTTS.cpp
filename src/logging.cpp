

#include "logging.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#endif

#if defined(_MSC_VER)
#include <chrono>  // NOLINT
#else
#include <sys/time.h>
#include <unistd.h>
#endif

namespace melo {
namespace logging {

LoggingWrapper::LogLevel LoggingWrapper::log_level_cfg_ =
    LoggingWrapper::LogLevel::MELO_INFO;

void LoggingWrapper::LogSetLevel(LogLevel Level) { log_level_cfg_ = Level; }

LoggingWrapper::LogLevel LoggingWrapper::LogGetLevel() {
  return log_level_cfg_;
}

LoggingWrapper::~LoggingWrapper() {
  if (log_level_cfg_ == LoggingWrapper::LogLevel::MELO_OFF) {
    return;
  }

  if (log_level_cfg_ <= log_level_) {
#if defined(_WIN32)
    char sep = '\\';
#else
    char sep = '/';
#endif
    const char* const partial_name = strrchr(filename_, sep);
    std::stringstream ss;
    ss << "TDIWEF"[static_cast<int>(log_level_)] << ' '
       << (partial_name != nullptr ? partial_name + 1 : filename_) << ':'
       << line_ << "] " << stream_.str();

#if defined(ANDROID) || defined(__ANDROID__)
    int android_log_level = ANDROID_LOG_INFO;
    switch (log_level_) {
      case LogLevel::MELO_TRACE:
        android_log_level = ANDROID_LOG_VERBOSE;
        break;
      case LogLevel::MELO_DEBUG:
        android_log_level = ANDROID_LOG_DEBUG;
        break;
      case LogLevel::MELO_INFO:
        android_log_level = ANDROID_LOG_INFO;
        break;
      case LogLevel::MELO_WARN:
        android_log_level = ANDROID_LOG_WARN;
        break;
      case LogLevel::MELO_ERROR:
        android_log_level = ANDROID_LOG_ERROR;
        break;
      case LogLevel::MELO_FATAL:
        android_log_level = ANDROID_LOG_FATAL;
        break;
      case LogLevel::MELO_OFF:
        (void)android_log_level;
        break;
      default: {
        (void)android_log_level;
        break;
      }
    }
    __android_log_write(android_log_level, "FaceUnity-FUAI", ss.str().c_str());
#else
  std::cerr << ss.str() << std::endl;
#endif

    if (log_level_ == LogLevel::MELO_FATAL) {
      std::flush(std::cerr);
      std::abort();
    }
  }
}

std::string LogLevelToStr(LoggingWrapper::LogLevel severity) {
  switch (severity) {
    case LoggingWrapper::LogLevel::MELO_TRACE:
      return "trace";
    case LoggingWrapper::LogLevel::MELO_DEBUG:
      return "debug";
    case LoggingWrapper::LogLevel::MELO_INFO:
      return "info";
    case LoggingWrapper::LogLevel::MELO_WARN:
      return "warn";
    case LoggingWrapper::LogLevel::MELO_ERROR:
      return "error";
    case LoggingWrapper::LogLevel::MELO_FATAL:
      return "fatal";
    case LoggingWrapper::LogLevel::MELO_OFF:
      return "off";
    default:
        return "";
  }
}

LoggingWrapper::LogLevel LogLevelFromStr(const std::string& str) {
  if (str == "trace") {
    return LoggingWrapper::LogLevel::MELO_TRACE;
  } else if (str == "debug") {
    return LoggingWrapper::LogLevel::MELO_DEBUG;
  } else if (str == "info") {
    return LoggingWrapper::LogLevel::MELO_INFO;
  } else if (str == "warn") {
    return LoggingWrapper::LogLevel::MELO_WARN;
  } else if (str == "error") {
    return LoggingWrapper::LogLevel::MELO_ERROR;
  } else if (str == "fatal") {
    return LoggingWrapper::LogLevel::MELO_FATAL;
  } else if (str == "off") {
    return LoggingWrapper::LogLevel::MELO_OFF;
  } else {
    return LoggingWrapper::LogLevel::MELO_INFO;
  }
}

Status RecordErrorLogInfo(const char* file_name, const int line,
                          const std::string& msg) {
  std::string msg_tmp = "[";
  msg_tmp += MELO_FILE_NAME(file_name);
  msg_tmp += ":";
  msg_tmp += std::to_string(line);
  msg_tmp += " ] ";
  msg_tmp += msg;
  return melo::Status(melo::error::MELO_UNKNOWN, msg_tmp);
}

void RecordLogInfo(const char* file_name, const int line, std::string& msg) {
  std::string MELO_tmp = "[";
  MELO_tmp += MELO_FILE_NAME(file_name);
  MELO_tmp += ":";
  MELO_tmp += std::to_string(line);
  MELO_tmp += "] ";
  MELO_tmp += msg;
  MELO_tmp += "\n";
  msg = MELO_tmp;
}

}  // namespace logging

static bool is_debug_image = false;
bool IsDebugImage() { return is_debug_image; }
void SetDebugImage(bool enable) { is_debug_image = enable; }

void DumpMemory(const char* file, const char* fun, int line, int& native_o,
                int& graphics_o) {
#if defined(ANDROID) || defined(__ANDROID__)
  // if (!g_dump_memory_enable) {
  //   return;
  // }
  int pid = static_cast<int>(getpid());
  std::string cmd = "dumpsys meminfo ";
  cmd += std::to_string(pid);
  std::cout << "dump memory: file:" << file << " function:" << fun
            << " line:" << line << std::endl;
  // system(cmd.c_str());

  FILE* pf;
  pf = popen(cmd.c_str(), "r");
  int native_heap = -1;
  int graphics = -1;
  if (pf != NULL) {
    char buf[1024];
    while (fgets(buf, 1024, pf) != NULL) {
      // printf("get popen %s\n",buf);
      char* native_heap_str = strstr(buf, "Native Heap:");
      if (native_heap_str != NULL) {
        native_heap = atoi(native_heap_str + sizeof("Native Heap:"));
        // printf("get native_heap %d\n",native_heap);
      }
      char* graphics_str = strstr(buf, "Graphics:");
      if (graphics_str != NULL) {
        graphics = atoi(graphics_str + sizeof("Graphics:"));
        // printf("get graphics %d\n",graphics);
      }
    }
    pclose(pf);
    pf = NULL;
  }
  if (native_heap != -1 || graphics != -1) {
    std::cout << "dump memory: native_heap(KB):" << native_heap
              << " graphics(KB):" << graphics << std::endl;
  }

  native_o = native_heap;
  graphics_o = graphics;

#endif
}

}  // namespace melo
