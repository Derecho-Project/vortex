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

#define MY_UUID "2bc87cc7-7ade-43d1-9a47-2a497b2fd5c0"
#define MY_DESC "Demo DLL UDL: on each message, build 256x256 GPU tensors and call Python add_gpu."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

struct GlobalState {
	std::unique_ptr<pyscheduler::PyManager> python = nullptr;
	std::unique_ptr<pyscheduler::PyManager::InvokeHandler> invoke_d = nullptr;
	int my_task_id = 0;
};

std::unique_ptr<GlobalState> globals = nullptr;

class TaskD_OCDPO : public VortexWorkerUdl {
protected:
	void initialize_resources() override {
		using pyscheduler::PyManager;
		spdlog::info("[d_udl] initialize_resources start");
		globals = std::make_unique<GlobalState>();
		globals->python = std::make_unique<PyManager>();
		globals->python->add_path("python_udls");
		globals->python->add_path("/home/yy354/.local/lib/python3.10/site-packages");
		globals->invoke_d = std::make_unique<PyManager::InvokeHandler>(
			globals->python->loadPythonModule("step_a", "invoke"));
		globals->my_task_id = _registry->find_task_id_by_path("/D").value_or(3);
		spdlog::info("[d_udl] initialize_resources done task_id={}", globals->my_task_id);
	}

	void execute_udl(scheduler::TaskBinding binding,
					 DefaultCascadeContextType* typed_ctxt,
					 uint32_t worker_id) override {
		(void)typed_ctxt;
		(void)worker_id;

		std::size_t b_count = 0;
		std::size_t c_count = 0;
		std::size_t total_payload_bytes = 0;
		for(const auto& handle : binding.inputs) {
			auto payload_span = _join_service->resolve(handle);
			if(!payload_span) {
				continue;
			}

			auto msgB = StepBMessage::from_bytes(
				nullptr, reinterpret_cast<const uint8_t*>(payload_span->data()));
			if(msgB) {
				++b_count;
				total_payload_bytes += msgB->message.size();
				continue;
			}

			auto msgC = StepCMessage::from_bytes(
				nullptr, reinterpret_cast<const uint8_t*>(payload_span->data()));
			if(msgC) {
				++c_count;
				total_payload_bytes += msgC->message.size();
			}
		}

		report_completion(binding, worker_id, typed_ctxt);
		spdlog::info("[d_udl] job={} READY b_inputs={} c_inputs={} total_payload_bytes={}",
					 binding.task.job_id,
					 b_count,
					 c_count,
					 total_payload_bytes);
	}

public:
	TaskD_OCDPO()
		: VortexWorkerUdl("TaskD_OCDPO", "/D", "/scheduleD", MY_UUID) { }

	VORTEX_DEFINE_CLASS_METHODS(TaskD_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskD_OCDPO)
} // namespace cascade
} // namespace derecho
