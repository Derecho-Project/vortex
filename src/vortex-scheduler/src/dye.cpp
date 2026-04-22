#include <vortex_scheduler/dye.hpp>

#include <chrono>

namespace scheduler::timestamp {
namespace {
std::uint64_t task_ref_to_uint64(const message::TaskRef& task_ref) {
	return static_cast<std::uint64_t>(task_ref.graph_id) << 48 | static_cast<std::uint64_t>(task_ref.job_id) << 32 |
		   static_cast<std::uint64_t>(task_ref.task_id) << 16 | static_cast<std::uint64_t>(task_ref.stream_id);
}

std::filesystem::path with_timestamp_suffix(const std::filesystem::path& base) {
	const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
	auto out = base;
	const auto stem = base.stem().string();
	const auto ext = base.extension().string();
	out.replace_filename(stem + "." + std::to_string(now_ns) + ext);
	return out;
}
} // namespace

void DyeLogger::log_ingress(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept {
	derecho::cascade::TimestampLogger::log(
		DYE_TAG, node_id, static_cast<std::uint64_t>(MessageId::INGRESS), task_ref_to_uint64(task_ref));
}

void DyeLogger::log_join(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept {
	derecho::cascade::TimestampLogger::log(
		DYE_TAG, node_id, static_cast<std::uint64_t>(MessageId::JOIN), task_ref_to_uint64(task_ref));
}

void DyeLogger::log_execution(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept {
	derecho::cascade::TimestampLogger::log(
		DYE_TAG, node_id, static_cast<std::uint64_t>(MessageId::EXECUTION), task_ref_to_uint64(task_ref));
}

void DyeLogger::log_holding(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept {
	derecho::cascade::TimestampLogger::log(
		DYE_TAG, node_id, static_cast<std::uint64_t>(MessageId::HOLDING), task_ref_to_uint64(task_ref));
}

void DyeLogger::log_egress(std::uint64_t node_id, const message::TaskRef& task_ref) noexcept {
	derecho::cascade::TimestampLogger::log(
		DYE_TAG, node_id, static_cast<std::uint64_t>(MessageId::EGRESS), task_ref_to_uint64(task_ref));
}

void DyeLogger::flush(const std::filesystem::path& file_path) noexcept {
	derecho::cascade::TimestampLogger::flush(with_timestamp_suffix(file_path));
}

} // namespace scheduler::timestamp
