#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cascade/cascade.hpp>
#include <cascade/object.hpp>
#include <cascade/service_client_api.hpp>
#include <cascade/user_defined_logic_interface.hpp>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

#include <vortex_scheduler/dag_registry.hpp>
#include <vortex_scheduler/messages.hpp>
#include <vortex_scheduler/serde.hpp>
#include <vortex_scheduler/udl_base.hpp>

namespace derecho {
namespace cascade {

#define MY_UUID "3e4d73a1-0fe6-4f53-8d19-2e5f7333a001"
#define MY_DESC "Scheduler UDL: receives WorkerStatus messages and emits SchedulerCommand decisions."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

namespace {

/// @brief view a blob as an immutable byte span (zero-copy).
std::span<const std::byte> as_byte_span(const Blob& blob) {
	return std::span<const std::byte>(reinterpret_cast<const std::byte*>(blob.bytes), blob.size);
}

} // namespace

class Scheduler_OCDPO : public OffCriticalDataPathObserver {
private:
	std::unique_ptr<scheduler::DagRegistry> _registry;
	std::once_flag _init_flag;

	void initialize_resources() {
		// Empty UUID loads all graphs (the scheduler is not bound to a single task UDL).
		_registry = std::make_unique<scheduler::DagRegistry>(
			std::filesystem::path("jobs.json"), std::string_view{});
	}

public:
	void operator()(const derecho::node_id_t sender,
					const std::string& key_string,
					const uint32_t prefix_length,
					persistent::version_t version,
					const mutils::ByteRepresentable* const value_ptr,
					const std::unordered_map<std::string, bool>& outputs,
					ICascadeContext* ctxt,
					uint32_t worker_id) override {
		(void)sender;
		(void)prefix_length;
		(void)version;
		(void)outputs;
		(void)worker_id;

		std::call_once(_init_flag, [this]() { initialize_resources(); });

		const auto obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
		if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
			return;
		}

		auto* typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
		if(typed_ctxt == nullptr) {
			spdlog::error("scheduler_udl: cascade context is not DefaultCascadeContextType");
			return;
		}

		scheduler::message::WorkerStatusView status;
		try {
			status = scheduler::message::WorkerStatusView::from_buffer(as_byte_span(obj->blob));
		} catch(const std::exception& e) {
			spdlog::warn("scheduler_udl: failed to decode WorkerStatus from key '{}': {}",
						 key_string, e.what());
			return;
		}

		// Group decisions by source pathname so each source worker DLL receives
		// a single SchedulerCommand on its own scheduler_path (e.g. "/A/SCHED").
		std::unordered_map<std::string, std::vector<scheduler::message::Decision>> by_source_path;

		for(const auto& completed : status.completed) {
			const auto* src_node = _registry->find_node(completed.task_id);
			if(src_node == nullptr) {
				spdlog::warn("scheduler_udl: completed task_id {} not in registry", completed.task_id);
				continue;
			}

			const auto downstream = _registry->get_downstream_tasks(
				completed.graph_id, completed.task_id);

			if(downstream.empty()) {
				continue; // sink task -- no routing needed
			}

			auto& bucket = by_source_path[src_node->pathname + "/SCHED"];
			bucket.reserve(bucket.size() + downstream.size());

			for(const auto downstream_task_id : downstream) {
				const scheduler::message::TaskRef target_ref{
					.graph_id  = completed.graph_id,
					.job_id    = completed.job_id,
					.task_id   = downstream_task_id,
					.stream_id = completed.stream_id,
				};

				bucket.push_back(scheduler::message::Decision{
					.source_worker = status.worker,
					.target_worker = scheduler::message::WorkerRef{ .shard_number = 0 },
					.source_task   = completed,
					.target_task   = target_ref,
				});
			}
		}

		auto& capi = typed_ctxt->get_service_client_ref();

		for(auto& [scheduler_path_prefix, decisions] : by_source_path) {
			scheduler::message::SchedulerCommand command{
				.decisions     = std::move(decisions),
				.cancellations = {},
			};

			for (const auto& decision : command.decisions) {
				spdlog::info("[sched]: [{}]: ({}, {}, {}, {}) => ({}, {}, {}, {}) @ {} => {}",
							 scheduler_path_prefix,
							 decision.source_task.graph_id,
							 decision.source_task.job_id,
							 decision.source_task.task_id,
							 decision.source_task.stream_id,
							 decision.target_task.graph_id,
							 decision.target_task.job_id,
							 decision.target_task.task_id,
							 decision.target_task.stream_id,
							 decision.source_worker.shard_number,
							 decision.target_worker.shard_number
							);
			}

			const std::size_t total_size = command.size_estimate();

			auto blob = Blob(
				[c = std::move(command)](std::uint8_t* out, std::size_t cap) -> std::size_t {
					const auto out_span = std::span<std::byte>(reinterpret_cast<std::byte*>(out), cap);
					return c.to_buffer(out_span);
				},
				total_size);

			const std::string out_key = scheduler_path_prefix + "/"
				+ std::to_string(status.worker.shard_number);
			ObjectWithStringKey out_obj(out_key, std::move(blob));

			try {
				capi.template put_and_forget<VolatileCascadeStoreWithStringKey>(
					out_obj, /*subgroup_index*/ 0, /*shard_index*/ 0, true);
			} catch(const std::exception& e) {
				spdlog::error("scheduler_udl: failed to publish SchedulerCommand to '{}': {}",
							  out_key, e.what());
			}
		}
	}

	VORTEX_DEFINE_CLASS_METHODS(Scheduler_OCDPO)
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(Scheduler_OCDPO)

} // namespace cascade
} // namespace derecho
