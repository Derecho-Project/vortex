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
	std::unique_ptr<scheduler::TaskJoinService> join_service = nullptr;
	int my_task_id;
};

std::unique_ptr<GlobalState> globals = nullptr;

class TaskA_OCDPO : public OffCriticalDataPathObserver {
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
		(void)value_ptr;
		(void)outputs;
		(void)ctxt;

		static int job_count = 0;

		// lazy initialization of global state
		if(!globals) [[unlikely]] {
			globals = std::make_unique<GlobalState>();
			globals->python = std::make_unique<pyscheduler::PyManager>();
			globals->python->add_path("python_udls");
			globals->python->add_path("/home/yy354/.local/lib/python3.10/site-packages");
			globals->invoke_a = std::make_unique<pyscheduler::PyManager::InvokeHandler>(
				globals->python->loadPythonModule("step_a", "invoke"));
			globals->join_service = std::make_unique<scheduler::TaskJoinService>(
				scheduler::DagRegistry::from_dfg_file("jobs.json", MY_UUID));
			globals->my_task_id =
				globals->join_service->dag().find_task_id_by_path("/A").value_or(0);

			spdlog::info("[a_udl] initialized");
		}

		auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);

		StepAMessage msg;
		msg.message = "Hello world from step A!";

		scheduler::TaskOutput to;
		to.worker_id = worker_id;
		to.job_id = job_count++;
		to.target_task_id = globals->my_task_id;
		to.source_task_id = globals->my_task_id;
		to.graph_id = 0;
		to.payload_size = static_cast<uint32_t>(msg.size_estimate());

		const auto header_size = to.size_estimate();
		const auto total_size = header_size + to.payload_size;

		auto make_blob = [=](const scheduler::TaskOutput& header) {
			return Blob(
				[header, msg, header_size](uint8_t* out, std::size_t) mutable -> std::size_t {
					header.to_bytes(out);
					msg.to_bytes(out + header_size);
					return header_size + msg.size_estimate();
				},
				total_size);
		};

		for(const auto downstream_task :
			globals->join_service->dag().find_task(0, globals->my_task_id)->downstream) {
			scheduler::TaskOutput to_local = to;
			to_local.target_task_id = downstream_task;

			spdlog::info(
				"[a_udl] emit job={} -> task={} msg='{}'", to.job_id, downstream_task, msg.message);

			ObjectWithStringKey obj;
			obj.key = globals->join_service->dag().find_task(0, downstream_task)->pathname + "/" +
					  std::to_string(to.job_id);
			obj.blob = make_blob(to_local);

			typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(
				obj, 0, 0, true);
		}
	}

	static void initialize() {
		if(!ocdpo_ptr) {
			ocdpo_ptr = std::make_shared<TaskA_OCDPO>();
		}
	}

	static std::shared_ptr<OffCriticalDataPathObserver> get() {
		return ocdpo_ptr;
	}

private:
	static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskA_OCDPO)
} // namespace cascade
} // namespace derecho
