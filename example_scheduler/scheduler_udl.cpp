#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cascade/object.hpp>
#include <cascade/user_defined_logic_interface.hpp>
#include <spdlog/logger.h>
#include <vortex_scheduler/prelude.hpp>

#include "base.hpp"

namespace derecho {
namespace cascade {

#define MY_UUID "3e4d73a1-0fe6-4f53-8d19-2e5f7333a001"
#define MY_DESC "Scheduler UDL: receives trigger messages and emits SchedulerCommand control messages."
VORTEX_DEFINE_UDL_METADATA(MY_UUID, MY_DESC)

class Scheduler_OCDPO : public OffCriticalDataPathObserver {
private:
	static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

	void send_scheduler_command(DefaultCascadeContextType* typed_ctxt,
							const std::string& schedule_path,
							uint16_t job_id,
							const scheduler::SchedulerCommand& cmd) {
		if(!typed_ctxt) {
			return;
		}

		const size_t cmd_size = cmd.size_estimate();
		ObjectWithStringKey out_obj;
		out_obj.key = schedule_path + "/" + std::to_string(job_id);
		out_obj.blob = Blob(
			[cmd](uint8_t* out, std::size_t) mutable -> std::size_t {
				cmd.to_bytes(out);
				return cmd.size_estimate();
			},
			cmd_size);

		typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(
			out_obj, 0, 0, true);

		spdlog::info("[scheduler_udl] dispatched command key={} decisions={} cancellations={}",
					 out_obj.key,
					 cmd.decisions.size(),
					 cmd.cancellations.size());
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

		auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
		if(!typed_ctxt) {
			spdlog::error("[scheduler_udl] empty cascade context");
			return;
		}

		const auto* obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
		if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
			spdlog::warn("[scheduler_udl] empty trigger object");
			return;
		}

		auto status = scheduler::WorkerStatus::from_bytes(nullptr, obj->blob.bytes);
		if(!status) {
			spdlog::warn("[scheduler_udl] invalid WorkerStatus payload on key={}", key_string);
			return;
		}

		spdlog::info("[scheduler_udl] status received key={} worker={} completed={}",
					 key_string,
					 status->worker_id,
					 status->completed.size());

		for(const auto& done : status->completed) {
			scheduler::SchedulerCommand cmd_a {
				std::vector<scheduler::Decision> {},
				std::vector<scheduler::TaskRef> {},
			};
			scheduler::SchedulerCommand cmd_b {
				std::vector<scheduler::Decision> {},
				std::vector<scheduler::TaskRef> {},
			};
			scheduler::SchedulerCommand cmd_c {
				std::vector<scheduler::Decision> {},
				std::vector<scheduler::TaskRef> {},
			};

			if(done.task_id == 0) {
				cmd_a.decisions.emplace_back(
					status->worker_id, 1, scheduler::TaskRef {done.graph_id, done.job_id, 0});
				cmd_a.decisions.emplace_back(
					status->worker_id, 2, scheduler::TaskRef {done.graph_id, done.job_id, 0});
				send_scheduler_command(typed_ctxt, "/scheduleA", done.job_id, cmd_a);
			}

			if(done.task_id == 1) {
				cmd_b.decisions.emplace_back(
					status->worker_id, 3, scheduler::TaskRef {done.graph_id, done.job_id, 1});
				send_scheduler_command(typed_ctxt, "/scheduleB", done.job_id, cmd_b);
			}

			if(done.task_id == 2) {
				cmd_c.decisions.emplace_back(
					status->worker_id, 3, scheduler::TaskRef {done.graph_id, done.job_id, 2});
				send_scheduler_command(typed_ctxt, "/scheduleC", done.job_id, cmd_c);
			}

			spdlog::info("[scheduler_udl] processed completion graph={} job={} task={}",
					 done.graph_id,
					 done.job_id,
					 done.task_id);
		}
	}

	static void initialize() {
		if(!ocdpo_ptr) {
			ocdpo_ptr = std::make_shared<Scheduler_OCDPO>();
		}
	}

	static std::shared_ptr<OffCriticalDataPathObserver> get() {
		return ocdpo_ptr;
	}
};

VORTEX_DEFINE_UDL_ENTRYPOINTS(Scheduler_OCDPO)

} // namespace cascade
} // namespace derecho
