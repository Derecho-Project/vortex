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

#define MY_UUID "24e10f1c-1100-11eb-1111-0111ac110002"
#define MY_DESC "Demo DLL UDL: on each message, build 256x256 GPU tensors and call Python add_gpu."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

struct GlobalState {
	std::unique_ptr<pyscheduler::PyManager> python = nullptr;
	std::unique_ptr<pyscheduler::PyManager::InvokeHandler> invoke_a = nullptr;
	int my_task_id;
};

std::unique_ptr<GlobalState> globals = nullptr;

class TaskA_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override {
		using pyscheduler::PyManager;

		globals = std::make_unique<GlobalState>();
		globals->python = std::make_unique<PyManager>();
		globals->python->add_path("python_udls");
		globals->python->add_path("/home/yy354/.local/lib/python3.10/site-packages");
		globals->invoke_a = std::make_unique<PyManager::InvokeHandler>(
			globals->python->loadPythonModule("step_a", "invoke"));
	}

	void execute_udl(scheduler::TaskBinding binding) override { 
		// _join_service->resolve();
	};

public:
	TaskA_OCDPO()
		: VortexWorkerUdl("TaskA_OCDPO", "/A", "/scheduleA", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskA_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskA_OCDPO)
} // namespace cascade
} // namespace derecho


			// std::string combined;
			// for(const auto& handle : binding->inputs) {
			// 	auto payload_span = globals->join_service->resolve(handle);
			// 	if(!payload_span) {
			// 		continue;
			// 	}
			// 	// Attempt to parse as StepB then StepC
			// 	auto msgB = StepBMessage::from_bytes(nullptr,
			// 		reinterpret_cast<const uint8_t*>(payload_span->data()));
			// 	if(msgB) {
			// 		combined += "[B:" + msgB->message + "] ";
			// 		continue;
			// 	}
			// 	auto msgC = StepCMessage::from_bytes(nullptr,
			// 		reinterpret_cast<const uint8_t*>(payload_span->data()));
			// 	if(msgC) {
			// 		combined += "[C:" + msgC->message + "] ";
			// 	}
			// }

			// spdlog::info("[d_udl] job={} READY inputs={}", binding->task.job_id, combined);
