#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vortex_scheduler/core.hpp>
#include <vortex_scheduler/ingress_types.hpp>
#include <vortex_scheduler/messages.hpp>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

class JoinTable {
public:
	JoinTable() = default;

	std::optional<TaskBinding> add_input(const TaskRef& task,
										 uint16_t processor_id,
										 uint16_t expected_inputs,
										 uint16_t dependency_slot,
										 const BlobHandle& payload);

	void clear_task(const TaskRef& task);

	std::size_t pending_tasks() const;

private:
	struct TaskKey {
		uint16_t job_id;
		uint16_t task_id;

		bool operator==(const TaskKey& other) const {
			return job_id == other.job_id && task_id == other.task_id;
		}
	};

	struct TaskKeyHash {
		std::size_t operator()(const TaskKey& key) const {
			return (static_cast<std::size_t>(key.job_id) << 16) ^
				   static_cast<std::size_t>(key.task_id);
		}
	};

	struct State {
		uint16_t processor_id = 0;
		uint16_t expected_inputs = 0;
		uint16_t received = 0;
		std::vector<BlobHandle> inputs;
		std::vector<uint8_t> present;
	};

	std::unordered_map<TaskKey, State, TaskKeyHash> states_;
};

VORTEX_SCHEDULER_NAMESPACE_END
