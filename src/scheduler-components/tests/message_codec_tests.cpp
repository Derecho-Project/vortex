#include <catch2/catch.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <derecho/mutils-serialization/SerializationSupport.hpp>

#include <vortex_scheduler/messages.hpp>

VORTEX_SCHEDULER_IMPORT

namespace {

template <typename T>
std::unique_ptr<T> roundtrip(const T& value) {
  const std::size_t size = value.size_estimate();
  std::vector<uint8_t> buffer(size);
  const std::size_t written = value.to_bytes(buffer.data());
  REQUIRE(written == size);

  auto decoded = T::from_bytes(nullptr, buffer.data());
  REQUIRE(decoded != nullptr);
  return decoded;
}

}  // namespace

TEST_CASE("TaskRef roundtrip serialization", "[messages][codec]") {
  TaskRef original{0, 42, 7};
  auto decoded = roundtrip(original);
  CHECK(decoded->job_id == original.job_id);
  CHECK(decoded->task_id == original.task_id);
}

TEST_CASE("SchedulerCommand roundtrip serialization", "[messages][codec]") {
  SchedulerCommand original{
      std::vector<Decision>{Decision{3, 9, TaskRef{0, 100, 11}},
                            Decision{4, 10, TaskRef{0, 100, 12}}},
      std::vector<TaskRef>{TaskRef{0, 100, 1}, TaskRef{0, 100, 2}}};

  auto decoded = roundtrip(original);
  REQUIRE(decoded->decisions.size() == 2);
  CHECK(decoded->decisions[0].worker_id == 3);
  CHECK(decoded->decisions[0].target_task_id == 9);
  CHECK(decoded->decisions[0].task.job_id == 100);
  CHECK(decoded->decisions[0].task.task_id == 11);
  CHECK(decoded->decisions[1].worker_id == 4);
  CHECK(decoded->decisions[1].target_task_id == 10);
  CHECK(decoded->cancellations.size() == 2);
  CHECK(decoded->cancellations[0].job_id == 100);
  CHECK(decoded->cancellations[0].task_id == 1);
}

TEST_CASE("WorkerStatus roundtrip serialization", "[messages][codec]") {
  WorkerStatus original{5,
                        17,
                        3,
                        123456,
                        std::vector<TaskRef>{TaskRef{0, 9, 1}, TaskRef{0, 9, 2}}};

  auto decoded = roundtrip(original);
  CHECK(decoded->worker_id == 5);
  CHECK(decoded->task_id == 17);
  CHECK(decoded->queue_size == 3);
  CHECK(decoded->time_to_empty == 123456);
  REQUIRE(decoded->completed.size() == 2);
  CHECK(decoded->completed[0].graph_id == 0);
  CHECK(decoded->completed[0].job_id == 9);
  CHECK(decoded->completed[0].task_id == 1);
  CHECK(decoded->completed[1].task_id == 2);
}

TEST_CASE("TaskOutputHeader roundtrip serialization", "[messages][codec]") {
  TaskOutput original{8, 77, 6, 5, 9, 13};

  auto decoded = roundtrip(original);
  CHECK(decoded->worker_id == 8);
  CHECK(decoded->job_id == 77);
  CHECK(decoded->target_task_id == 6);
  CHECK(decoded->source_task_id == 5);
  CHECK(decoded->graph_id == 9);
  CHECK(decoded->payload_size == 13);
}

TEST_CASE("TaskOutput payload span is runtime-only", "[messages][codec]") {
  const std::vector<std::byte> payload{
      std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
  const TaskOutput original(
      3,
      99,
      7,
      4,
      2,
      payload.size(),
      std::span<const std::byte>(payload.data(), payload.size()));
  const TaskOutput header_only(3, 99, 7, 4, 2, payload.size());

  REQUIRE(original.payload.size() == payload.size());
  CHECK(original.size_estimate() == header_only.size_estimate());

  auto decoded = roundtrip(original);
  CHECK(decoded->worker_id == 3);
  CHECK(decoded->job_id == 99);
  CHECK(decoded->target_task_id == 7);
  CHECK(decoded->source_task_id == 4);
  CHECK(decoded->graph_id == 2);
  CHECK(decoded->payload_size == payload.size());

  CHECK(decoded->payload.empty());
}
