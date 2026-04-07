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
	std::unique_ptr<pyscheduler::PyManager::InvokeHandler> invoke_a = nullptr;
	std::unique_ptr<scheduler::TaskJoinService> join_service = nullptr;
	int my_task_id = 0;
};

std::unique_ptr<GlobalState> globals = nullptr;

class TaskB_OCDPO : public OffCriticalDataPathObserver {
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
				globals->my_task_id = globals->join_service->dag().find_task_id_by_path("/B").value_or(1);

				spdlog::info("[b_udl] initialized");
			}

			auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
			if(!typed_ctxt) {
				return;
			}

			const auto* obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
			if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
				return;
			}

			const uint8_t* buf = obj->blob.bytes;
			auto in_header = scheduler::TaskOutput::from_bytes(nullptr, buf);
			const auto header_size = in_header->size_estimate();
			const uint8_t* payload_ptr = buf + header_size;

			auto in_msg = StepAMessage::from_bytes(nullptr, payload_ptr);

			spdlog::info("[b_udl] recv job={} from task={} msg='{}'",
						 in_header->job_id, in_header->target_task_id, in_msg->message);

			StepBMessage out_msg{std::string("B saw: ") + in_msg->message};

			scheduler::TaskOutput to;
			to.worker_id = worker_id;
			to.job_id = in_header->job_id;
			to.target_task_id = globals->my_task_id;
			to.source_task_id = globals->my_task_id;
			to.graph_id = in_header->graph_id;
			to.payload_size = static_cast<uint32_t>(out_msg.size_estimate());

			const auto out_header_size = to.size_estimate();
			const auto total_size = out_header_size + to.payload_size;

			auto make_blob = [&](const scheduler::TaskOutput& header) {
				return Blob(
					[header, out_msg, out_header_size](uint8_t* out, std::size_t) mutable -> std::size_t {
						header.to_bytes(out);
						out_msg.to_bytes(out + out_header_size);
						return out_header_size + out_msg.size_estimate();
					},
					total_size);
			};

			for (const auto downstream_task : globals->join_service->dag().find_task(0, globals->my_task_id)->downstream) {
				scheduler::TaskOutput to_local = to;
				to_local.target_task_id = downstream_task;

				spdlog::info("[b_udl] emit job={} -> task={} msg='{}'",
							 to.job_id, downstream_task, out_msg.message);

				ObjectWithStringKey out_obj;
				out_obj.key = globals->join_service->dag().find_task(0, downstream_task)->pathname + "/" + std::to_string(to.job_id);
				out_obj.blob = make_blob(to_local);

				typed_ctxt->get_service_client_ref()
					.put_and_forget<VolatileCascadeStoreWithStringKey>(out_obj, 0, 0, true);
			}
		}

	static void initialize() {
		if(!ocdpo_ptr) {
			ocdpo_ptr = std::make_shared<TaskB_OCDPO>();
		}
	}

	static std::shared_ptr<OffCriticalDataPathObserver> get() {
		return ocdpo_ptr;
	}

private:
	static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(TaskB_OCDPO)
} // namespace cascade
} // namespace derecho
