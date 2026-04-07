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
	std::unique_ptr<pyscheduler::PyManager::InvokeHandler> invoke_a = nullptr;
	std::unique_ptr<scheduler::TaskJoinService> join_service = nullptr;
	int my_task_id = 0;
};

std::unique_ptr<GlobalState> globals = nullptr;

class TaskD_OCDPO : public OffCriticalDataPathObserver {
public:
	void operator()(const derecho::node_id_t sender,
					const std::string& key_string,
					const uint32_t prefix_length,
					persistent::version_t version,
					const mutils::ByteRepresentable* const value_ptr,
					const std::unordered_map<std::string, bool>& outputs,
					ICascadeContext* ctxt,
					uint32_t worker_id) override {
			(void)version;
			(void)outputs;

			if(!globals) [[unlikely]] {
				globals = std::make_unique<GlobalState>();
				globals->python = std::make_unique<pyscheduler::PyManager>();
				globals->python->add_path("python_udls");
				globals->python->add_path("/home/yy354/.local/lib/python3.10/site-packages");
				globals->invoke_a = std::make_unique<pyscheduler::PyManager::InvokeHandler>(
					globals->python->loadPythonModule("step_a", "invoke"));
				globals->join_service = std::make_unique<scheduler::TaskJoinService>(
					scheduler::DagRegistry::from_dfg_file("jobs.json", MY_UUID));
				globals->my_task_id = globals->join_service->dag().find_task_id_by_path("/D").value_or(3);

				spdlog::info("[d_udl] initialized");
			}

			auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
			if(!typed_ctxt) {
				return;
			}

			const auto* obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
			if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
				return;
			}

			std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(obj->blob.bytes),
											 obj->blob.size);

			auto binding = globals->join_service->recv(key_string, bytes);
			if(!binding) {
				return;
			}

			std::string combined;
			for(const auto& handle : binding->inputs) {
				auto payload_span = globals->join_service->resolve(handle);
				if(!payload_span) {
					continue;
				}
				// Attempt to parse as StepB then StepC
				auto msgB = StepBMessage::from_bytes(nullptr,
					reinterpret_cast<const uint8_t*>(payload_span->data()));
				if(msgB) {
					combined += "[B:" + msgB->message + "] ";
					continue;
				}
				auto msgC = StepCMessage::from_bytes(nullptr,
					reinterpret_cast<const uint8_t*>(payload_span->data()));
				if(msgC) {
					combined += "[C:" + msgC->message + "] ";
				}
			}

			spdlog::info("[d_udl] job={} READY inputs={}", binding->task.job_id, combined);

			// terminal stage: no downstream emit
			volatile int *x = new int[10];
			x = (int*) malloc(123);
		}

	static void initialize() {
		if(!ocdpo_ptr) {
			ocdpo_ptr = std::make_shared<TaskD_OCDPO>();
		}
	}

	static std::shared_ptr<OffCriticalDataPathObserver> get() {
		return ocdpo_ptr;
	}

private:
	static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskD_OCDPO)
} // namespace cascade
} // namespace derecho
