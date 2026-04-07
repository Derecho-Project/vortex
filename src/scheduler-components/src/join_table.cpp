#include <vortex_scheduler/join_table.hpp>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

std::optional<TaskBinding> JoinTable::add_input(const TaskRef& task,
                                                uint16_t processor_id,
                                                uint16_t expected_inputs,
                                                uint16_t dependency_slot,
                                                const BlobHandle& payload) {
  if(expected_inputs == 0) {
    throw std::invalid_argument("expected_inputs must be greater than zero");
  }
  if(dependency_slot >= expected_inputs) {
    throw std::out_of_range("dependency_slot is out of range for expected_inputs");
  }

  const TaskKey key{task.job_id, task.task_id};
  auto [it, inserted] = states_.emplace(key, State{});
  State& state = it->second;

  if(inserted) {
    state.processor_id = processor_id;
    state.expected_inputs = expected_inputs;
    state.inputs.resize(expected_inputs);
    state.present.assign(expected_inputs, 0);
    state.received = 0;
  } else {
    if(state.processor_id != processor_id) {
      throw std::invalid_argument("processor_id mismatch for existing task state");
    }
    if(state.expected_inputs != expected_inputs) {
      throw std::invalid_argument("expected_inputs mismatch for existing task state");
    }
  }

  if(state.present[dependency_slot]) {
    return std::nullopt;
  }

  state.inputs[dependency_slot] = payload;
  state.present[dependency_slot] = 1;
  ++state.received;

  if(state.received < state.expected_inputs) {
    return std::nullopt;
  }

  TaskBinding binding{task, state.processor_id, std::move(state.inputs)};

  states_.erase(it);
  return binding;
}

void JoinTable::clear_task(const TaskRef& task) {
  states_.erase(TaskKey{task.job_id, task.task_id});
}

std::size_t JoinTable::pending_tasks() const {
  return states_.size();
}

VORTEX_SCHEDULER_NAMESPACE_END
