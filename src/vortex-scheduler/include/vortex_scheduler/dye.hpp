/** Wrapper over the TimestampLogger class in Derecho Cascade to log timestamps of various events in the scheduler.
 *
 */

#pragma once
#include "messages.hpp"
#include <cascade/utils.hpp>
#include <cstdint>
#include <filesystem>

namespace scheduler::timestamp {

class DyeLogger {
public:
	DyeLogger() = delete;

	/// @brief record when the "source task" arrives at the worker node
	static void log_ingress(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept;

	/// @brief recorded when the "target task" is emitted from the task join service
	static void log_join(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept;

	/// @brief recorded when the "target task" starts execution
	static void log_execution(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept;

	/// @brief log when the "target task" finishes execution and is emitted to a holding queue
	static void log_holding(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept;

	/// @brief log when the "target task" is sent to the next udl
	static void log_egress(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept;

	/// @brief flush all logged timestamps to a file
	static void flush(const std::filesystem::path& file_path) noexcept;

private:
	static constexpr std::uint64_t DYE_TAG = 255;
	enum struct MessageId : std::uint64_t {
		INGRESS = 1,
		JOIN = 2,
		EXECUTION = 4,
		HOLDING = 5,
		EGRESS = 6,
	};
};

} // namespace scheduler::timestamp
