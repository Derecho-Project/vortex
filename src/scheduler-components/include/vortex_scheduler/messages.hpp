/** Control and Data messages sent between worker and scheduler nodes
 *
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cascade/object.hpp>
#include <derecho/mutils-serialization/SerializationSupport.hpp>
#include <type_traits>
#include <vortex_scheduler/core.hpp>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

enum class MessageClass : uint8_t {
	Control = 1,
	Data = 2,
};

/// @brief default implementation of serialization
template <typename Derived>
class VortexSerde {
public:
	size_t size_estimate() const noexcept {
		static_assert(std::is_base_of_v<mutils::ByteRepresentable, Derived>,
					  "VortexSerde requires the derived type to satisfy ByteRepresentable");
		return static_cast<const Derived&>(*this).bytes_size();
	}

	size_t to_bytes(uint8_t* out) const noexcept {
		static_assert(std::is_base_of_v<mutils::ByteRepresentable, Derived>,
					  "VortexSerde requires the derived type to satisfy ByteRepresentable");
		return mutils::to_bytes(static_cast<const Derived&>(*this), out);
	}

	void post_object(const std::function<void(uint8_t const* const, std::size_t)>& f) const {
		static_assert(std::is_base_of_v<mutils::ByteRepresentable, Derived>,
					  "VortexSerde requires the derived type to satisfy ByteRepresentable");
		mutils::post_object(f, static_cast<const Derived&>(*this));
	}

	static std::unique_ptr<Derived> from_bytes(mutils::DeserializationManager* dm,
											   const uint8_t* buf) {
		static_assert(std::is_base_of_v<mutils::ByteRepresentable, Derived>,
					  "VortexSerde requires the derived type to satisfy ByteRepresentable");
		return mutils::from_bytes<Derived>(dm, buf);
	}

	static mutils::context_ptr<Derived> from_bytes_noalloc(mutils::DeserializationManager* dm,
														   const uint8_t* buf) {
		static_assert(std::is_base_of_v<mutils::ByteRepresentable, Derived>,
					  "VortexSerde requires the derived type to satisfy ByteRepresentable");
		return mutils::from_bytes_noalloc<Derived>(dm, buf);
	}

	static mutils::context_ptr<const Derived>
	from_bytes_noalloc_const(mutils::DeserializationManager* dm, const uint8_t* buf) {
		static_assert(std::is_base_of_v<mutils::ByteRepresentable, Derived>,
					  "VortexSerde requires the derived type to satisfy ByteRepresentable");
		return mutils::from_bytes_noalloc<const Derived>(dm, buf);
	}
};

struct TaskRef : public mutils::ByteRepresentable, public VortexSerde<TaskRef> {
	uint16_t graph_id;
	uint16_t job_id;
	uint16_t task_id;

	TaskRef() = delete;
	TaskRef(uint16_t graph_id, uint16_t job_id, uint16_t task_id)
		: graph_id(graph_id)
		, job_id(job_id)
		, task_id(task_id) { }
	DEFAULT_SERIALIZATION_SUPPORT(TaskRef, graph_id, job_id, task_id);
};

struct Decision : public mutils::ByteRepresentable, public VortexSerde<Decision> {
	uint16_t worker_id = 0;
	uint16_t target_task_id = 0;
	TaskRef task;

	Decision() = default;
	Decision(uint16_t worker_id, uint16_t target_task_id, TaskRef task)
		: worker_id(worker_id)
		, target_task_id(target_task_id)
		, task(std::move(task)) { }

	DEFAULT_SERIALIZATION_SUPPORT(Decision, worker_id, target_task_id, task);
};

struct SchedulerCommand : public mutils::ByteRepresentable, public VortexSerde<SchedulerCommand> {
	std::vector<Decision> decisions;
	std::vector<TaskRef> cancellations;

	SchedulerCommand() = default;
	SchedulerCommand(std::vector<Decision> decisions, std::vector<TaskRef> cancellations)
		: decisions(std::move(decisions))
		, cancellations(std::move(cancellations)) { }

	DEFAULT_SERIALIZATION_SUPPORT(SchedulerCommand, decisions, cancellations);
};

struct WorkerStatus : public mutils::ByteRepresentable, public VortexSerde<WorkerStatus> {
	uint16_t worker_id = 0;
	uint16_t task_id = 0;
	uint16_t queue_size = 0;
	uint64_t time_to_empty = 0;
	std::vector<TaskRef> completed;

	WorkerStatus() = default;
	WorkerStatus(uint16_t worker_id,
				 uint16_t task_id,
				 uint16_t queue_size,
				 uint64_t time_to_empty,
				 std::vector<TaskRef> completed)
		: worker_id(worker_id)
		, task_id(task_id)
		, queue_size(queue_size)
		, time_to_empty(time_to_empty)
		, completed(std::move(completed)) { }

	DEFAULT_SERIALIZATION_SUPPORT(
		WorkerStatus, worker_id, task_id, queue_size, time_to_empty, completed);
};

struct TaskOutput : public mutils::ByteRepresentable, public VortexSerde<TaskOutput> {
	uint16_t worker_id;
	uint16_t job_id;
	uint16_t target_task_id;
	uint16_t source_task_id;
	uint16_t graph_id = 0;
	uint32_t payload_size = 0;
	std::span<const std::byte> payload = { }; // runtime-only payload view; not serialized

	TaskOutput() = default;
	TaskOutput(uint16_t worker_id,
			   uint16_t job_id,
			   uint16_t target_task_id,
			   uint16_t source_task_id,
			   uint16_t graph_id = 0,
			   uint32_t payload_size = 0,
			   std::span<const std::byte> payload = { })
		: worker_id(worker_id)
		, job_id(job_id)
		, target_task_id(target_task_id)
		, source_task_id(source_task_id)
		, graph_id(graph_id)
		, payload_size(payload_size)
		, payload(payload) { }

	DEFAULT_SERIALIZATION_SUPPORT(TaskOutput, worker_id, job_id, target_task_id, source_task_id, graph_id, payload_size);
};

VORTEX_SCHEDULER_NAMESPACE_END
