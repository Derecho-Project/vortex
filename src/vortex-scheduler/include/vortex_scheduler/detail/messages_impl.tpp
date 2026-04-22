#pragma once
// Implementation of templated message serializers from messages.hpp.
// Included at the bottom of messages.hpp; not a standalone header.

#include <cassert>
#include <cstdint>

#include <vortex_scheduler/serde.hpp>

namespace scheduler::message {

namespace detail {
/// @brief Tag identifying which message variant is in the wire format.
enum struct MessageComponent : std::uint8_t {
	SCHEDULER_COMMAND = 0,
	WORKER_STATUS     = 1,
	TASK_OUTPUT       = 2,
};
} // namespace detail

// ~~~ SchedulerCommandT ~~~

template <typename Family>
SchedulerCommandT<Family> SchedulerCommandT<Family>::from_buffer(
	std::span<const std::byte> payload) {
	serde::Reader reader{ payload };

	const auto type = reader.get<detail::MessageComponent>();
	assert(type == detail::MessageComponent::SCHEDULER_COMMAND);
	(void)type;

	const auto num_decisions     = reader.get<std::size_t>();
	const auto num_cancellations = reader.get<std::size_t>();

	const auto decisions_span     = reader.get_span<Decision>(num_decisions);
	const auto cancellations_span = reader.get_span<TaskRef>(num_cancellations);

	return SchedulerCommandT<Family>{
		.decisions     = decisions_repr::make(decisions_span),
		.cancellations = cancellations_repr::make(cancellations_span),
	};
}

template <typename Family>
std::size_t SchedulerCommandT<Family>::to_buffer(
	const std::span<std::byte>& buffer) const noexcept {
	serde::Writer writer{ buffer };

	const auto decisions_span     = decisions_repr::view(decisions);
	const auto cancellations_span = cancellations_repr::view(cancellations);

	writer.put(detail::MessageComponent::SCHEDULER_COMMAND);
	writer.put(decisions_span.size());
	writer.put(cancellations_span.size());
	writer.put_span(decisions_span);
	writer.put_span(cancellations_span);
	return writer.bytes_written();
}

template <typename Family>
std::size_t SchedulerCommandT<Family>::size_estimate() const noexcept {
	serde::Sizer sizer;

	const auto decisions_span     = decisions_repr::view(decisions);
	const auto cancellations_span = cancellations_repr::view(cancellations);

	sizer.put<detail::MessageComponent>();
	sizer.put<std::size_t>();
	sizer.put<std::size_t>();
	sizer.put_span(decisions_span);
	sizer.put_span(cancellations_span);
	return sizer.bytes_written();
}

// ~~~ WorkerStatusT ~~~

template <typename Family>
WorkerStatusT<Family> WorkerStatusT<Family>::from_buffer(
	std::span<const std::byte> payload) {
	serde::Reader reader{ payload };

	const auto type = reader.get<detail::MessageComponent>();
	assert(type == detail::MessageComponent::WORKER_STATUS);
	(void)type;

	const auto worker     = reader.get<WorkerRef>();
	const auto task_id    = reader.get<std::uint16_t>();
	const auto queue_size = reader.get<std::size_t>();
	const auto empty_ns   = reader.get<std::int64_t>();

	const auto num_completed   = reader.get<std::size_t>();
	const auto completed_span  = reader.get_span<TaskRef>(num_completed);

	return WorkerStatusT<Family>{
		.worker     = worker,
		.task_id    = task_id,
		.queue_size = queue_size,
		.empty_ns   = empty_ns,
		.completed  = completed_repr::make(completed_span),
	};
}

template <typename Family>
std::size_t WorkerStatusT<Family>::to_buffer(
	const std::span<std::byte>& buffer) const noexcept {
	serde::Writer writer{ buffer };

	const auto completed_span = completed_repr::view(completed);

	writer.put(detail::MessageComponent::WORKER_STATUS);
	writer.put(worker);
	writer.put(task_id);
	writer.put(queue_size);
	writer.put(empty_ns);
	writer.put(completed_span.size());
	writer.put_span(completed_span);
	return writer.bytes_written();
}

template <typename Family>
std::size_t WorkerStatusT<Family>::size_estimate() const noexcept {
	serde::Sizer sizer;

	const auto completed_span = completed_repr::view(completed);

	sizer.put<detail::MessageComponent>();
	sizer.put<WorkerRef>();
	sizer.put<std::uint16_t>();
	sizer.put<std::size_t>();
	sizer.put<std::int64_t>();
	sizer.put<std::size_t>();
	sizer.put_span(completed_span);
	return sizer.bytes_written();
}

// ~~~ TaskOutputT ~~~

template <typename Family>
TaskOutputT<Family> TaskOutputT<Family>::from_buffer(
	std::span<const std::byte> payload) {
	serde::Reader reader{ payload };

	const auto type = reader.get<detail::MessageComponent>();
	assert(type == detail::MessageComponent::TASK_OUTPUT);
	(void)type;

	const auto decision     = reader.get<Decision>();
	const auto payload_size = reader.get<std::size_t>();
	const auto payload_span = reader.get_span<std::byte>(payload_size);

	return TaskOutputT<Family>{
		.decision = decision,
		.payload  = payload_repr::make(payload_span),
	};
}

template <typename Family>
std::size_t TaskOutputT<Family>::to_buffer(
	const std::span<std::byte>& buffer) const noexcept {
	serde::Writer writer{ buffer };

	const auto payload_span = payload_repr::view(payload);

	writer.put(detail::MessageComponent::TASK_OUTPUT);
	writer.put(decision);
	writer.put(payload_span.size());
	writer.put_span(payload_span);
	return writer.bytes_written();
}

template <typename Family>
std::size_t TaskOutputT<Family>::size_estimate() const noexcept {
	serde::Sizer sizer;

	const auto payload_span = payload_repr::view(payload);

	sizer.put<detail::MessageComponent>();
	sizer.put<Decision>();
	sizer.put<std::size_t>();
	sizer.put_span(payload_span);
	return sizer.bytes_written();
}

} // namespace scheduler::message
