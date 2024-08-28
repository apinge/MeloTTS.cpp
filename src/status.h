
#ifndef MELO_COMMON_STATUS_H_
#define MELO_COMMON_STATUS_H_

#include <memory>
#include <string>

namespace melo {

    namespace error {

        enum Code {
            // Not an error; returned on success
            MELO_OK = 0,

            // Unknown error.
            MELO_UNKNOWN = 1,

            // Client specified an invalid argument.
            MELO_INVALID_ARGUMENT = 2,

            // Some requested entity (e.g., file or directory) was not found.
            MELO_NOT_FOUND = 5,

            // Some entity that we attempted to create (e.g., file or directory)
            // already exists.
            MELO_ALREADY_EXISTS = 6,

            // Operation is not implemented.
            MELO_UNIMPLEMENTED = 12,
        };

    }  // namespace error

    class Status {
    public:
        Status() {}

        explicit Status(error::Code code, const std::string message = "") {
            state_ = std::unique_ptr<State>(new State);
            state_->code = code;
            state_->message = message;
        }

        inline Status(const Status& s)
            : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

        void operator=(const Status& s) {
            if (s.state_ == nullptr) {
                state_ = nullptr;
            }
            else {
                state_ = std::unique_ptr<State>(new State(*s.state_));
            }
        }

        bool ok() const { return state_ == nullptr; }

        operator bool() const { return ok(); }

        static Status OK() { return Status(); }

        std::string message() const { return ok() ? "" : state_->message; }

        error::Code code() { return state_->code; }

    private:
        struct State {
            error::Code code;
            std::string message;
        };

        std::unique_ptr<State> state_;
    };

    enum MELO_ERROR_MSG_INDEX {
        MELO_NOT_IMPLEMENTED_ERROR = 0,
        MELO_INIT_MODEL_ERROR = 1,
        MELO_ERROR_MSG_INDEX_MAX = 2,
    };

    static inline const char* GetErrorMsg(MELO_ERROR_MSG_INDEX index) {
        switch (index) {
        case MELO_NOT_IMPLEMENTED_ERROR:
            return "Not implemented error!";
            break;
        case MELO_INIT_MODEL_ERROR:
            return "Init model error!";
            break;
        case MELO_ERROR_MSG_INDEX_MAX:
        default: {
            return nullptr;
            break;
        }
        }
    }

}  // namespace melo

#endif  // MELO_COMMON_STATUS_H_
