
#ifndef MELO_COMMON_LOGGING_H_
#define MELO_COMMON_LOGGING_H_

#include <string.h>
#include <sstream>
#include <string>
#include "status.h"

namespace melo {
namespace logging {

// A wrapper that logs to stderr.
class LoggingWrapper {
 public:
  enum class LogLevel : int {
    MELO_TRACE = 0,  //  print debug information inside the loop
    MELO_DEBUG = 1,  //  print debug information
    MELO_INFO = 2,   //  displays the program call process
    MELO_WARN = 3,   //  warning, but can continue running
    MELO_ERROR = 4,  //  causes an error
    MELO_FATAL = 5,  //  crash program
    MELO_OFF = 6,    //  close log
  };
  LoggingWrapper(const char* filename, int line, LogLevel log_level)
      : log_level_(log_level), filename_(filename), line_(line) {}
  std::stringstream& Stream() { return stream_; }
  ~LoggingWrapper();

  static void LogSetLevel(LogLevel severity);
  static LogLevel LogGetLevel();

 private:
  std::stringstream stream_;
  LogLevel log_level_;
  static LogLevel log_level_cfg_;

  const char* filename_;
  int line_;
};

std::string LogLevelToStr(LoggingWrapper::LogLevel severity);
LoggingWrapper::LogLevel LogLevelFromStr(const std::string& str);

Status RecordErrorLogInfo(const char* file_name, const int line,
                          const std::string& msg);
void RecordLogInfo(const char* file_name, const int line, std::string& msg);

}  // namespace logging

bool IsDebugImage();
void SetDebugImage(bool enable);

void DumpMemory(const char* file, const char* fun, int line, int& native_o,
                int& graphics_o);

}  // namespace melo

#define MELO_LOG(log_level)                                                   \
  melo::logging::LoggingWrapper(                                              \
      __FILE__, __LINE__, melo::logging::LoggingWrapper::LogLevel::log_level) \
      .Stream()

#define MELO_LOG_IF(log_level, condition) \
  if (condition) MELO_LOG(log_level)  // NOLINT

#define MELO_LOG_IS_ON(log_level)                  \
  (melo::logging::LoggingWrapper::LogGetLevel() <= \
   melo::logging::LoggingWrapper::LogLevel::log_level)  // NOLINT

#define MELO_LOG_SET_LEVEL(log_level)         \
  melo::logging::LoggingWrapper::LogSetLevel( \
      melo::logging::LoggingWrapper::LogLevel::log_level);

#define MELO_LOG_SET_LEVEL_BY_STRING(str)                  \
  {                                                        \
    melo::logging::LoggingWrapper::LogLevel log_level =    \
        melo::logging::LogLevelFromStr(str);               \
    melo::logging::LoggingWrapper::LogSetLevel(log_level); \
  }

/////////////////// check
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#define MELO_FILE_NAME(x) strrchr(x, '\\') ? strrchr(x, '\\') + 1 : x
#else
#define MELO_FILE_NAME(x) strrchr(x, '/') ? strrchr(x, '/') + 1 : x
#endif

// #define MELO_DEBUG
#ifdef MELO_DEBUG
#define MELO_CHECK(condition) \
  MELO_LOG_IF(MELO_FATAL, !(condition)) << "Check failed: (" #condition ") "
#else
#define MELO_CHECK(condition)                                          \
  if (!(condition)) {                                                  \
    std::string msg = "data check fail";                               \
    MELO_LOG(MELO_ERROR) << msg;                                       \
    return melo::logging::RecordErrorLogInfo(__FILE__, __LINE__, msg); \
  }
#endif

#define MELO_CHECK_EQ(a, b) MELO_CHECK((a) == (b))
#define MELO_CHECK_NE(a, b) MELO_CHECK((a) != (b))
#define MELO_CHECK_LE(a, b) MELO_CHECK((a) <= (b))
#define MELO_CHECK_LT(a, b) MELO_CHECK((a) < (b))
#define MELO_CHECK_GE(a, b) MELO_CHECK((a) >= (b))
#define MELO_CHECK_GT(a, b) MELO_CHECK((a) > (b))

#define MELO_CHECK_CRASH(condition) \
  MELO_LOG_IF(MELO_FATAL, !(condition)) << "Check failed: (" #condition ") "

#define MELO_RETURN_IF_ERROR(...)               \
  do {                                          \
    const melo::Status _status = (__VA_ARGS__); \
    if (!_status.ok()) {                        \
      return _status;                           \
    }                                           \
  } while (0)

#define MELO_RECORD_MSG(msg) \
  melo::logging::RecordLogInfo(__FILE__, __LINE__, msg);

#define MELO_ERROR_RETURN(msg) \
  return melo::logging::RecordErrorLogInfo(__FILE__, __LINE__, msg);

#endif  // MELO_COMMON_LOGGING_H_
