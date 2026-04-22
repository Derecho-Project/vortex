/** Control and Data messages sent between worker and scheduler nodes */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <cascade/object.hpp>
#include <unordered_map>

#include <vortex_scheduler/serde.hpp>

// communication namespace for the scheduler
namespace scheduler::message {

/// @brief uniquely identifies a task
struct TaskRef {
	/// @brief identifies which execution graph this job belongs to
	std::uint16_t graph_id;

	/// @brief unique job identifier (strict total order within a stream)
	std::uint16_t job_id;

	/// @brief specifies the specific task in the job execution graph identified by graph_id
	std::uint16_t task_id;

	/// @brief tasks within a stream are linearizable
	std::uint16_t stream_id;
};

/// @brief uniquely identifies a worker
struct WorkerRef {
	/// @brief shard number of this worker
	std::uint16_t shard_number;
};

/// @brief issued from the scheduler to the sending worker node
struct Decision {
	/// @brief current worker
	WorkerRef source_worker;

	/// @brief desired recipient worker
	WorkerRef target_worker;

	/// @brief current task at the sending worker node
	TaskRef source_task;

	/// @brief desired task
	TaskRef target_task;
};

// ~~~ Messages ~~~
//
// Each message is parameterized on a serde::Family:
//   - serde::ViewFamily   -> non-owning views (std::span). Zero-copy; valid
//                            only for the lifetime of the source buffer.
//   - serde::OwnedFamily  -> owning copies (std::vector). Allocates.
//
// Type aliases at the bottom expose `*View` (non-owning) and the bare name
// (owning), e.g. `SchedulerCommandView` and `SchedulerCommand`.

template <typename Family>
struct SchedulerCommandT {
	using decisions_repr     = serde::array_repr_t<Family, Decision>;
	using cancellations_repr = serde::array_repr_t<Family, TaskRef>;

	typename decisions_repr::type     decisions;
	typename cancellations_repr::type cancellations;

	/// @note  For ViewFamily, returned spans alias `payload`; valid only for
	///        the lifetime of `payload`.
	/// @note  For OwnedFamily, allocates and may throw std::bad_alloc.
	[[nodiscard]] static SchedulerCommandT<Family> from_buffer(std::span<const std::byte> payload);
	[[nodiscard]] std::size_t to_buffer(const std::span<std::byte>& buffer) const noexcept;
	[[nodiscard]] std::size_t size_estimate() const noexcept;
};

template <typename Family>
struct WorkerStatusT {
	using completed_repr = serde::array_repr_t<Family, TaskRef>;

	WorkerRef     worker;
	std::uint16_t task_id;
	std::size_t   queue_size;
	std::int64_t  empty_ns;
	typename completed_repr::type completed;

	[[nodiscard]] static WorkerStatusT<Family> from_buffer(std::span<const std::byte> payload);
	[[nodiscard]] std::size_t to_buffer(const std::span<std::byte>& buffer) const noexcept;
	[[nodiscard]] std::size_t size_estimate() const noexcept;
};

template <typename Family>
struct TaskOutputT {
	using payload_repr = serde::array_repr_t<Family, std::byte>;

	Decision decision;
	typename payload_repr::type payload;

	[[nodiscard]] static TaskOutputT<Family> from_buffer(std::span<const std::byte> payload);
	[[nodiscard]] std::size_t to_buffer(const std::span<std::byte>& buffer) const noexcept;
	[[nodiscard]] std::size_t size_estimate() const noexcept;
};

using SchedulerCommandView = SchedulerCommandT<serde::ViewFamily>;
using SchedulerCommand     = SchedulerCommandT<serde::OwnedFamily>;

using WorkerStatusView     = WorkerStatusT<serde::ViewFamily>;
using WorkerStatus         = WorkerStatusT<serde::OwnedFamily>;

using TaskOutputView       = TaskOutputT<serde::ViewFamily>;
using TaskOutput           = TaskOutputT<serde::OwnedFamily>;

} // namespace scheduler::message

#include <vortex_scheduler/detail/messages_impl.tpp>

namespace scheduler::message {
static_assert(serde::BufferSerializable<SchedulerCommandView>);
static_assert(serde::BufferSerializable<SchedulerCommand>);
static_assert(serde::BufferSerializable<WorkerStatusView>);
static_assert(serde::BufferSerializable<WorkerStatus>);
static_assert(serde::BufferSerializable<TaskOutputView>);
static_assert(serde::BufferSerializable<TaskOutput>);
} // namespace scheduler::message
