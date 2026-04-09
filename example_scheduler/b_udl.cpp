#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include <cascade/object.hpp>

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <pybind11/pytypes.h>
#include <pyscheduler/pyscheduler.hpp>
#include <pyscheduler/tensor.hpp>
#include <spdlog/logger.h>

#include <vortex_scheduler/prelude.hpp>

#include "base.hpp"
#include "diamond_messages.hpp"

namespace derecho {
namespace cascade {

#define MY_UUID "a945d81b-53c7-43c3-b461-12c85d6883ab"
#define MY_DESC "Demo DLL UDL: on each message, build 256x256 GPU tensors and call Python add_gpu."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

struct GlobalState {
	std::unique_ptr<pyscheduler::PyManager> python = nullptr;
	std::unique_ptr<pyscheduler::PyManager::InvokeHandler> invoke_b = nullptr;
	int my_task_id = 0;
};

std::unique_ptr<GlobalState> globals = nullptr;

class TaskB_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override {
		using pyscheduler::PyManager;
		spdlog::info("[b_udl] initialize_resources start");
		globals = std::make_unique<GlobalState>();
		globals->python = std::make_unique<PyManager>();
		globals->python->add_path("python_udls");
		globals->python->add_path("/home/yy354/.local/lib/python3.10/site-packages");
		globals->invoke_b = std::make_unique<PyManager::InvokeHandler>(
			globals->python->loadPythonModule("step_a", "invoke"));
		globals->my_task_id = _registry->find_task_id_by_path("/B").value_or(1);
		spdlog::info("[b_udl] initialize_resources done task_id={}", globals->my_task_id);
	}

	void execute_udl(scheduler::TaskBinding binding,
					 DefaultCascadeContextType* typed_ctxt,
					 uint32_t worker_id) override {
		if(binding.inputs.empty()) {
			spdlog::warn("[b_udl] job={} has no inputs", binding.task.job_id);
			return;
		}

		auto payload_span = _join_service->resolve(binding.inputs.front());
		if(!payload_span) {
			spdlog::warn("[b_udl] failed to resolve input payload for job={}", binding.task.job_id);
			return;
		}

		auto in_msg = StepAMessage::from_bytes(
			nullptr, reinterpret_cast<const uint8_t*>(payload_span->data()));
		if(!in_msg) {
			spdlog::warn("[b_udl] invalid StepAMessage for job={}", binding.task.job_id);
			return;
		}

		spdlog::info("[b_udl] execute job={} msg='{}'", binding.task.job_id, in_msg->message);
		StepBMessage out_msg {std::string("B saw: ") + in_msg->message};
		std::vector<std::byte> payload(out_msg.size_estimate());
		out_msg.to_bytes(reinterpret_cast<uint8_t*>(payload.data()));
		finish(binding, worker_id, std::move(payload), typed_ctxt);
		spdlog::info("[b_udl] execute done job={} (output queued)", binding.task.job_id);
	}

public:
	TaskB_OCDPO()
		: VortexWorkerUdl("TaskB_OCDPO", "/B", "/scheduleB", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskB_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskB_OCDPO)
} // namespace cascade
} // namespace derecho
