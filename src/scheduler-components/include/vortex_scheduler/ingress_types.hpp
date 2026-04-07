#pragma once

#include <cstdint>
#include <vector>

#include <vortex_scheduler/core.hpp>
#include <vortex_scheduler/messages.hpp>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

// Internal runtime handle for arena-owned payload bytes.
struct BlobHandle {
  uint8_t pool_class = 0;
  uint32_t segment_id = 0;
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct IngressRecord {
  TaskOutput header;
  uint16_t dependency_slot = 0;
  BlobHandle payload;
};

struct TaskBinding {
  TaskRef task;
  uint16_t processor_id = 0;
  std::vector<BlobHandle> inputs;

  TaskBinding() = default;
  TaskBinding(TaskRef task, uint16_t processor_id, std::vector<BlobHandle> inputs)
      : task(std::move(task)), processor_id(processor_id), inputs(std::move(inputs)) {}
};

struct TaskSpec {
  uint16_t expected_inputs = 0;
  uint16_t processor_id = 0;
  bool supports_batching = false;
};

VORTEX_SCHEDULER_NAMESPACE_END
