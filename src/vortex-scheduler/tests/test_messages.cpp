#include <catch2/catch.hpp>

#include <vortex_scheduler/messages.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace scheduler {
using namespace scheduler::message;

namespace {
template <typename T>
T roundtrip(const T& value, std::vector<std::byte>& buffer) {
	const std::size_t size_estimate = value.size_estimate();

	buffer.resize(size_estimate);
	std::span<std::byte> buffer_span(buffer.data(), buffer.size());

	const std::size_t written = value.to_buffer(buffer_span);
	REQUIRE(written == size_estimate);

	auto decoded = T::from_buffer(buffer_span);
	return decoded;
}

} // namespace

TEST_CASE("SchedulerCommandView roundtrip serialization", "[messages][codec]") {
	// NOTE: SchedulerCommandView holds std::span, so the backing vectors must
	// outlive the view. Keep them as named locals here.
	std::vector<Decision> decisions{
		Decision{ WorkerRef{ 3 }, WorkerRef{ 9 }, TaskRef{ 0, 100, 11 }, TaskRef{ } },
		Decision{ WorkerRef{ 4 }, WorkerRef{ 10 }, TaskRef{ 0, 100, 12 }, TaskRef{ } },
	};
	std::vector<TaskRef> cancellations{
		TaskRef{ 0, 100, 1 },
		TaskRef{ 0, 100, 2 },
	};
	SchedulerCommandView original{ decisions, cancellations };

	std::vector<std::byte> buffer;
	auto decoded = roundtrip(original, buffer);
	REQUIRE(decoded.decisions.size() == 2);

	CHECK(decoded.decisions[0].source_worker.shard_number == 3);
	CHECK(decoded.decisions[0].target_worker.shard_number == 9);
	CHECK(decoded.decisions[0].source_task.graph_id == 0);
	CHECK(decoded.decisions[0].source_task.job_id == 100);
	CHECK(decoded.decisions[0].source_task.task_id == 11);
	CHECK(decoded.decisions[1].source_worker.shard_number == 4);
	CHECK(decoded.decisions[1].target_worker.shard_number == 10);
	CHECK(decoded.decisions[1].source_task.graph_id == 0);
	CHECK(decoded.decisions[1].source_task.job_id == 100);
	CHECK(decoded.decisions[1].source_task.task_id == 12);
	CHECK(decoded.cancellations.size() == 2);
	CHECK(decoded.cancellations[0].graph_id == 0);
	CHECK(decoded.cancellations[0].job_id == 100);
	CHECK(decoded.cancellations[0].task_id == 1);
	CHECK(decoded.cancellations[1].graph_id == 0);
	CHECK(decoded.cancellations[1].job_id == 100);
	CHECK(decoded.cancellations[1].task_id == 2);
}

TEST_CASE("WorkerStatusView roundtrip serialization", "[messages][codec]") {
	WorkerRef worker{ 5 };
	std::vector<TaskRef> completed{
		TaskRef{ 0, 9, 1 },
		TaskRef{ 0, 9, 2 },
	};
	WorkerStatusView original{ worker, 17, 3, 0b1101101101, completed };

	std::vector<std::byte> buffer;
	auto decoded = roundtrip(original, buffer);
	CHECK(decoded.worker.shard_number == 5);
	CHECK(decoded.task_id == 17);
	CHECK(decoded.queue_size == 3);
	CHECK(decoded.empty_ns == 0b1101101101);
	REQUIRE(decoded.completed.size() == 2);
	CHECK(decoded.completed[0].graph_id == 0);
	CHECK(decoded.completed[0].job_id == 9);
	CHECK(decoded.completed[0].task_id == 1);
	CHECK(decoded.completed[1].graph_id == 0);
	CHECK(decoded.completed[1].job_id == 9);
	CHECK(decoded.completed[1].task_id == 2);
}

TEST_CASE("TaskOutputView roundtrip serialization", "[messages][codec]") {
	Decision decision{ WorkerRef{ 3 }, WorkerRef{ 9 }, TaskRef{ 0, 100, 11 }, TaskRef{ } };
	std::vector<std::byte> payload{ std::byte(0xDE), std::byte(0xAD), std::byte(0xBE), std::byte(0xEF) };
	TaskOutputView original{ decision, payload };

	std::vector<std::byte> buffer;
	auto decoded = roundtrip(original, buffer);
	CHECK(decoded.decision.source_worker.shard_number == 3);
	CHECK(decoded.decision.target_worker.shard_number == 9);
	CHECK(decoded.decision.source_task.graph_id == 0);
	CHECK(decoded.decision.source_task.job_id == 100);
	CHECK(decoded.decision.source_task.task_id == 11);
	REQUIRE(decoded.payload.size() == 4);
	CHECK(decoded.payload[0] == std::byte{ 0xDE });
	CHECK(decoded.payload[1] == std::byte{ 0xAD });
	CHECK(decoded.payload[2] == std::byte{ 0xBE });
	CHECK(decoded.payload[3] == std::byte{ 0xEF });
}

// ~~~ Fuzz tests ~~~
//
// Roundtrip fuzzing: generate random *valid* inputs (varying span sizes and
// field values, including edge cases like empty spans), serialize, then
// deserialize and verify field-by-field equality. Each TEST_CASE runs many
// iterations against a deterministic PRNG so failures are reproducible.

namespace {
constexpr int kFuzzIterations = 1'000;
constexpr std::size_t kMaxSpanSize = 1024;
constexpr std::uint64_t kFuzzSeed = 0xC0FFEEULL;

template <typename Rng>
TaskRef random_task_ref(Rng& rng) {
	std::uniform_int_distribution<std::uint32_t> dist(0, std::numeric_limits<std::uint16_t>::max());
	return TaskRef{
		static_cast<std::uint16_t>(dist(rng)),
		static_cast<std::uint16_t>(dist(rng)),
		static_cast<std::uint16_t>(dist(rng)),
		static_cast<std::uint16_t>(dist(rng)),
	};
}

template <typename Rng>
WorkerRef random_worker_ref(Rng& rng) {
	std::uniform_int_distribution<std::uint32_t> dist(0, std::numeric_limits<std::uint16_t>::max());
	return WorkerRef{ static_cast<std::uint16_t>(dist(rng)) };
}

template <typename Rng>
Decision random_decision(Rng& rng) {
	return Decision{
		random_worker_ref(rng),
		random_worker_ref(rng),
		random_task_ref(rng),
		random_task_ref(rng),
	};
}


template <typename Rng>
std::size_t random_span_size(Rng& rng) {
	// Bias towards including 0 and small sizes, but allow up to kMaxSpanSize.
	std::uniform_int_distribution<std::size_t> dist(0, kMaxSpanSize);
	return dist(rng);
}

bool task_ref_equal(const TaskRef& a, const TaskRef& b) {
	return a.graph_id == b.graph_id && a.job_id == b.job_id
		&& a.task_id == b.task_id && a.stream_id == b.stream_id;
}

bool decision_equal(const Decision& a, const Decision& b) {
	return a.source_worker.shard_number == b.source_worker.shard_number
		&& a.target_worker.shard_number == b.target_worker.shard_number
		&& task_ref_equal(a.source_task, b.source_task)
		&& task_ref_equal(a.target_task, b.target_task);
}

} // namespace

TEST_CASE("SchedulerCommandView fuzz roundtrip", "[messages][codec][fuzz]") {
	std::mt19937_64 rng(kFuzzSeed);
	std::vector<std::byte> buffer;

	for(int i = 0; i < kFuzzIterations; ++i) {
		std::vector<Decision> decisions(random_span_size(rng));
		for(auto& d : decisions) {
			d = random_decision(rng);
		}
		std::vector<TaskRef> cancellations(random_span_size(rng));
		for(auto& t : cancellations) {
			t = random_task_ref(rng);
		}

		SchedulerCommandView original{ decisions, cancellations };
		auto decoded = roundtrip(original, buffer);

		REQUIRE(decoded.decisions.size() == decisions.size());
		REQUIRE(decoded.cancellations.size() == cancellations.size());
		for(std::size_t k = 0; k < decisions.size(); ++k) {
			INFO("iteration=" << i << " decision_idx=" << k);
			CHECK(decision_equal(decoded.decisions[k], decisions[k]));
		}
		for(std::size_t k = 0; k < cancellations.size(); ++k) {
			INFO("iteration=" << i << " cancellation_idx=" << k);
			CHECK(task_ref_equal(decoded.cancellations[k], cancellations[k]));
		}
	}
}

TEST_CASE("WorkerStatusView fuzz roundtrip", "[messages][codec][fuzz]") {
	std::mt19937_64 rng(kFuzzSeed ^ 0x1);
	std::vector<std::byte> buffer;

	std::uniform_int_distribution<std::uint32_t> u16_dist(0, std::numeric_limits<std::uint16_t>::max());
	std::uniform_int_distribution<std::size_t> qsize_dist(0, std::numeric_limits<std::size_t>::max());
	std::uniform_int_distribution<std::int64_t> ns_dist(
		std::numeric_limits<std::int64_t>::min(),
		std::numeric_limits<std::int64_t>::max());

	for(int i = 0; i < kFuzzIterations; ++i) {
		WorkerRef worker = random_worker_ref(rng);
		auto task_id = static_cast<std::uint16_t>(u16_dist(rng));
		std::size_t queue_size = qsize_dist(rng);
		std::int64_t empty_ns = ns_dist(rng);

		std::vector<TaskRef> completed(random_span_size(rng));
		for(auto& t : completed) {
			t = random_task_ref(rng);
		}

		WorkerStatusView original{ worker, task_id, queue_size, empty_ns, completed };
		auto decoded = roundtrip(original, buffer);

		INFO("iteration=" << i);
		CHECK(decoded.worker.shard_number == worker.shard_number);
		CHECK(decoded.task_id == task_id);
		CHECK(decoded.queue_size == queue_size);
		CHECK(decoded.empty_ns == empty_ns);
		REQUIRE(decoded.completed.size() == completed.size());
		for(std::size_t k = 0; k < completed.size(); ++k) {
			INFO("completed_idx=" << k);
			CHECK(task_ref_equal(decoded.completed[k], completed[k]));
		}
	}
}

TEST_CASE("TaskOutputView fuzz roundtrip", "[messages][codec][fuzz]") {
	std::mt19937_64 rng(kFuzzSeed ^ 0x2);
	std::vector<std::byte> buffer;

	std::uniform_int_distribution<unsigned int> byte_dist(0, 255);

	for(int i = 0; i < kFuzzIterations; ++i) {
		Decision decision = random_decision(rng);

		std::vector<std::byte> payload(random_span_size(rng) * 4);
		for(auto& b : payload) {
			b = static_cast<std::byte>(byte_dist(rng));
		}

		TaskOutputView original{ decision, payload };
		auto decoded = roundtrip(original, buffer);

		INFO("iteration=" << i);
		CHECK(decision_equal(decoded.decision, decision));
		REQUIRE(decoded.payload.size() == payload.size());
		for(std::size_t k = 0; k < payload.size(); ++k) {
			INFO("payload_idx=" << k);
			CHECK(decoded.payload[k] == payload[k]);
		}
	}
}

// ~~~ Owning-variant tests ~~~
//
// The Owned* aliases (SchedulerCommand, WorkerStatus, TaskOutput) own their
// array fields via std::vector. These tests exercise:
//   1. Round-trip through their own to_buffer / from_buffer.
//   2. Cross-family interop: an OwnedFamily message must serialize identically
//      to its ViewFamily counterpart, and a ViewFamily message must
//      deserialize correctly into an OwnedFamily message (and vice versa).

TEST_CASE("SchedulerCommand owning roundtrip", "[messages][codec][owned]") {
	SchedulerCommand original{
		.decisions = {
			Decision{ WorkerRef{ 3 }, WorkerRef{ 9 }, TaskRef{ 0, 100, 11 }, TaskRef{ } },
			Decision{ WorkerRef{ 4 }, WorkerRef{ 10 }, TaskRef{ 0, 100, 12 }, TaskRef{ } },
		},
		.cancellations = {
			TaskRef{ 0, 100, 1 },
			TaskRef{ 0, 100, 2 },
		},
	};

	std::vector<std::byte> buffer;
	auto decoded = roundtrip(original, buffer);

	REQUIRE(decoded.decisions.size() == 2);
	CHECK(decision_equal(decoded.decisions[0], original.decisions[0]));
	CHECK(decision_equal(decoded.decisions[1], original.decisions[1]));
	REQUIRE(decoded.cancellations.size() == 2);
	CHECK(task_ref_equal(decoded.cancellations[0], original.cancellations[0]));
	CHECK(task_ref_equal(decoded.cancellations[1], original.cancellations[1]));

	// The decoded vector data lives independently of `buffer`: clearing the
	// buffer must not affect the owned message.
	buffer.clear();
	buffer.shrink_to_fit();
	CHECK(decoded.decisions.size() == 2);
	CHECK(decoded.cancellations.size() == 2);
}

TEST_CASE("WorkerStatus owning roundtrip", "[messages][codec][owned]") {
	WorkerStatus original{
		.worker = WorkerRef{ 5 },
		.task_id = 17,
		.queue_size = 3,
		.empty_ns = 0b1101101101,
		.completed = {
			TaskRef{ 0, 9, 1 },
			TaskRef{ 0, 9, 2 },
		},
	};

	std::vector<std::byte> buffer;
	auto decoded = roundtrip(original, buffer);

	CHECK(decoded.worker.shard_number == 5);
	CHECK(decoded.task_id == 17);
	CHECK(decoded.queue_size == 3);
	CHECK(decoded.empty_ns == 0b1101101101);
	REQUIRE(decoded.completed.size() == 2);
	CHECK(task_ref_equal(decoded.completed[0], original.completed[0]));
	CHECK(task_ref_equal(decoded.completed[1], original.completed[1]));
}

TEST_CASE("TaskOutput owning roundtrip", "[messages][codec][owned]") {
	TaskOutput original{
		.decision = Decision{ WorkerRef{ 3 }, WorkerRef{ 9 }, TaskRef{ 0, 100, 11 }, TaskRef{ } },
		.payload = std::vector<std::byte>{ std::byte(0xDE), std::byte(0xAD), std::byte(0xBE), std::byte(0xEF) },
	};

	std::vector<std::byte> buffer;
	auto decoded = roundtrip(original, buffer);

	CHECK(decision_equal(decoded.decision, original.decision));
	REQUIRE(decoded.payload.size() == original.payload.size());
	for(std::size_t k = 0; k < original.payload.size(); ++k) {
		CHECK(decoded.payload[k] == original.payload[k]);
	}
}

TEST_CASE("View<->Owned wire-format interop", "[messages][codec][owned]") {
	// Step 1: serialize an owning command, deserialize as a view, and confirm
	// the view aliases the buffer with matching data.
	SchedulerCommand owned{
		.decisions = {
			Decision{ WorkerRef{ 3 }, WorkerRef{ 9 }, TaskRef{ 0, 100, 11 }, TaskRef{ } },
		},
		.cancellations = {
			TaskRef{ 0, 100, 1 },
		},
	};

	std::vector<std::byte> buffer(owned.size_estimate());
	const std::size_t written = owned.to_buffer(buffer);
	REQUIRE(written == owned.size_estimate());

	auto view = SchedulerCommandView::from_buffer(buffer);
	REQUIRE(view.decisions.size() == owned.decisions.size());
	REQUIRE(view.cancellations.size() == owned.cancellations.size());
	CHECK(decision_equal(view.decisions[0], owned.decisions[0]));
	CHECK(task_ref_equal(view.cancellations[0], owned.cancellations[0]));

	// Step 2: the same buffer must also deserialize into an owning message.
	auto owned_again = SchedulerCommand::from_buffer(buffer);
	REQUIRE(owned_again.decisions.size() == owned.decisions.size());
	REQUIRE(owned_again.cancellations.size() == owned.cancellations.size());
	CHECK(decision_equal(owned_again.decisions[0], owned.decisions[0]));
	CHECK(task_ref_equal(owned_again.cancellations[0], owned.cancellations[0]));

	// Step 3: View and Owned must produce byte-identical wire output for
	// equivalent inputs (no padding leakage / ordering differences).
	SchedulerCommandView view_input{
		std::span<const Decision>(owned.decisions.data(), owned.decisions.size()),
		std::span<const TaskRef>(owned.cancellations.data(), owned.cancellations.size()),
	};
	std::vector<std::byte> view_buffer(view_input.size_estimate());
	const std::size_t view_written = view_input.to_buffer(view_buffer);
	REQUIRE(view_written == written);
	CHECK(std::memcmp(view_buffer.data(), buffer.data(), written) == 0);
}

TEST_CASE("SchedulerCommand owning fuzz roundtrip", "[messages][codec][owned][fuzz]") {
	std::mt19937_64 rng(kFuzzSeed ^ 0x10);
	std::vector<std::byte> buffer;

	for(int i = 0; i < kFuzzIterations; ++i) {
		SchedulerCommand original;
		original.decisions.resize(random_span_size(rng));
		for(auto& d : original.decisions) {
			d = random_decision(rng);
		}
		original.cancellations.resize(random_span_size(rng));
		for(auto& t : original.cancellations) {
			t = random_task_ref(rng);
		}

		auto decoded = roundtrip(original, buffer);

		REQUIRE(decoded.decisions.size() == original.decisions.size());
		REQUIRE(decoded.cancellations.size() == original.cancellations.size());
		for(std::size_t k = 0; k < original.decisions.size(); ++k) {
			INFO("iteration=" << i << " decision_idx=" << k);
			CHECK(decision_equal(decoded.decisions[k], original.decisions[k]));
		}
		for(std::size_t k = 0; k < original.cancellations.size(); ++k) {
			INFO("iteration=" << i << " cancellation_idx=" << k);
			CHECK(task_ref_equal(decoded.cancellations[k], original.cancellations[k]));
		}
	}
}

TEST_CASE("WorkerStatus owning fuzz roundtrip", "[messages][codec][owned][fuzz]") {
	std::mt19937_64 rng(kFuzzSeed ^ 0x11);
	std::vector<std::byte> buffer;

	std::uniform_int_distribution<std::uint32_t> u16_dist(0, std::numeric_limits<std::uint16_t>::max());
	std::uniform_int_distribution<std::size_t> qsize_dist(0, std::numeric_limits<std::size_t>::max());
	std::uniform_int_distribution<std::int64_t> ns_dist(
		std::numeric_limits<std::int64_t>::min(),
		std::numeric_limits<std::int64_t>::max());

	for(int i = 0; i < kFuzzIterations; ++i) {
		WorkerStatus original{
			.worker = random_worker_ref(rng),
			.task_id = static_cast<std::uint16_t>(u16_dist(rng)),
			.queue_size = qsize_dist(rng),
			.empty_ns = ns_dist(rng),
			.completed = {},
		};
		original.completed.resize(random_span_size(rng));
		for(auto& t : original.completed) {
			t = random_task_ref(rng);
		}

		auto decoded = roundtrip(original, buffer);

		INFO("iteration=" << i);
		CHECK(decoded.worker.shard_number == original.worker.shard_number);
		CHECK(decoded.task_id == original.task_id);
		CHECK(decoded.queue_size == original.queue_size);
		CHECK(decoded.empty_ns == original.empty_ns);
		REQUIRE(decoded.completed.size() == original.completed.size());
		for(std::size_t k = 0; k < original.completed.size(); ++k) {
			INFO("completed_idx=" << k);
			CHECK(task_ref_equal(decoded.completed[k], original.completed[k]));
		}
	}
}

TEST_CASE("TaskOutput owning fuzz roundtrip", "[messages][codec][owned][fuzz]") {
	std::mt19937_64 rng(kFuzzSeed ^ 0x12);
	std::vector<std::byte> buffer;

	std::uniform_int_distribution<unsigned int> byte_dist(0, 255);

	for(int i = 0; i < kFuzzIterations; ++i) {
		TaskOutput original;
		original.decision = random_decision(rng);
		original.payload.resize(random_span_size(rng) * 4);
		for(auto& b : original.payload) {
			b = static_cast<std::byte>(byte_dist(rng));
		}

		auto decoded = roundtrip(original, buffer);

		INFO("iteration=" << i);
		CHECK(decision_equal(decoded.decision, original.decision));
		REQUIRE(decoded.payload.size() == original.payload.size());
		for(std::size_t k = 0; k < original.payload.size(); ++k) {
			CHECK(decoded.payload[k] == original.payload[k]);
		}
	}
}

} // namespace scheduler