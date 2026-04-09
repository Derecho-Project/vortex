#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <condition_variable>
#include <spdlog/logger.h>

#include <cascade/object.hpp>
#include <cascade/user_defined_logic_interface.hpp>

#include <vortex_scheduler/decision_gate.hpp>
#include <vortex_scheduler/prelude.hpp>

// Generates the standard UDL DLL entrypoints for an OCDPO type.
// The OCDPO type must provide static initialize() and get() methods,
// and declare `static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;`.
#define VORTEX_DEFINE_UDL_ENTRYPOINTS(OCDPOType)                                                   \
	std::shared_ptr<OffCriticalDataPathObserver> OCDPOType::ocdpo_ptr;                             \
	void initialize(ICascadeContext* ctxt) {                                                       \
		(void)ctxt;                                                                                \
		OCDPOType::initialize();                                                                   \
	}                                                                                              \
	std::shared_ptr<OffCriticalDataPathObserver> get_observer(ICascadeContext* ctxt,               \
															  const nlohmann::json& cfg) {         \
		(void)ctxt;                                                                                \
		(void)cfg;                                                                                 \
		return OCDPOType::get();                                                                   \
	}                                                                                              \
	void release(ICascadeContext* ctxt) {                                                          \
		(void)ctxt;                                                                                \
	}

// Generates UUID/description exports from MY_UUID and MY_DESC.
// Define both MY_UUID and MY_DESC in the UDL .cpp before invoking this macro.
#define VORTEX_DEFINE_UDL_METADATA(uuid, desc)                                                     \
	std::string get_uuid() {                                                                       \
		return uuid;                                                                               \
	}                                                                                              \
	std::string get_description() {                                                                \
		return desc;                                                                               \
	}

#define VORTEX_DEFINE_CLASS_METHODS(OCDPOType)                                                     \
private:                                                                                           \
	static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;                                 \
                                                                                                   \
public:                                                                                            \
	static void initialize() {                                                                     \
		if(!ocdpo_ptr) {                                                                           \
			ocdpo_ptr = std::make_shared<OCDPOType>();                                             \
		}                                                                                          \
	}                                                                                              \
                                                                                                   \
	static std::shared_ptr<OffCriticalDataPathObserver> get() {                                    \
		return ocdpo_ptr;                                                                          \
	}

namespace derecho {
namespace cascade {

class VortexWorkerUdl : public OffCriticalDataPathObserver {
public:
	struct PendingPacket {
		std::string output_key;
		std::vector<std::byte> wire_packet;
	};

	VortexWorkerUdl() = delete;
	VortexWorkerUdl(const std::string_view& name,
					const std::string_view& data_path,
					const std::string_view& scheduler_path,
					const std::string_view& uuid)
		: _name(name)
		, _data_path(data_path)
		, _scheduler_path(scheduler_path)
		, _uuid(uuid) { }

	virtual ~VortexWorkerUdl() {
		{
			std::lock_guard<std::mutex> lock(_exec_mu);
			_exec_stop = true;
		}
		_exec_cv.notify_all();
		if(_exec_thread.joinable()) {
			_exec_thread.join();
		}
	}

private:
	struct ScopedPerfTimer {
		const std::string& worker_name;
		const char* label;
		std::chrono::steady_clock::time_point start;

		ScopedPerfTimer(const std::string& worker_name, const char* label)
			: worker_name(worker_name)
			, label(label)
			, start(std::chrono::steady_clock::now()) { }

		~ScopedPerfTimer() {
			const auto end = std::chrono::steady_clock::now();
			const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			spdlog::warn("[Worker:{}][perf] {} took {} us",
						 worker_name,
						 label,
						 elapsed.count());
		}
	};

	ScopedPerfTimer make_perf_timer(const char* label) const {
		return ScopedPerfTimer{ _name, label };
	}

	void log_perf_counter(const char* label, std::size_t value) const {
		spdlog::warn("[Worker:{}][perf] {}={}", _name, label, value);
	}

	void log_perf_counter(const char* label, uint32_t value) const {
		spdlog::warn("[Worker:{}][perf] {}={}", _name, label, value);
	}

	/// @brief name of this udl
	std::string _name;

	/// @brief path in which this udl receives data messages
	std::string _data_path;

	/// @brief path in which this udl receives scheduling messages
	std::string _scheduler_path;

	/// @brief uuid of of this udl that is not locked behind a compile time macro
	std::string _uuid;

	/// @brief initialization guard
	std::once_flag _init_flag;

protected:
	std::unique_ptr<scheduler::DagRegistry> _registry;
	std::unique_ptr<scheduler::TaskJoinService> _join_service;
	scheduler::DecisionGate<PendingPacket> _decision_gate;
	std::mutex _decision_gate_mu;

	std::mutex _exec_mu;
	std::condition_variable _exec_cv;
	bool _exec_has_work = false;
	bool _exec_stop = false;
	std::thread _exec_thread;
	DefaultCascadeContextType* _last_ctxt = nullptr;
	uint32_t _last_worker_id = 0;

protected:
	// methods which the child class should overload

	/// @brief called once the first time UDL receives data. purposes is to lazily load resource intensive computation units.
	virtual void initialize_resources() = 0;

	/// @brief dispatched for a task when upstream dependencies are satisfied
	virtual void execute_udl(scheduler::TaskBinding binding,
							 DefaultCascadeContextType* typed_ctxt,
							 uint32_t worker_id) = 0;

protected:
	void finish(const scheduler::TaskBinding& ingress_binding,
				uint32_t worker_id,
				std::vector<std::byte>&& payload,
				DefaultCascadeContextType* typed_ctxt) {
		auto perf_timer = make_perf_timer("finish");
		(void)perf_timer;

		if(!typed_ctxt || !_registry) {
			return;
		}

		const auto* src =
			_registry->find_task(ingress_binding.task.graph_id, ingress_binding.task.task_id);
		if(src == nullptr) {
			spdlog::warn(
				"[Worker:{}]: no DAG node for source task {}", _name, ingress_binding.task.task_id);
			return;
		}

		std::lock_guard<std::mutex> lock(_decision_gate_mu);
		log_perf_counter("finish.downstream_count", src->downstream.size());
		for(const uint16_t downstream_id : src->downstream) {
			const auto* dst = _registry->find_task(ingress_binding.task.graph_id, downstream_id);
			if(dst == nullptr) {
				continue;
			}

			scheduler::TaskOutput header;
			header.worker_id = static_cast<uint16_t>(worker_id);
			header.job_id = ingress_binding.task.job_id;
			header.target_task_id = downstream_id;
			header.source_task_id = ingress_binding.task.task_id;
			header.graph_id = ingress_binding.task.graph_id;
			header.payload_size = static_cast<uint32_t>(payload.size());

			const auto header_size = header.size_estimate();
			PendingPacket packet;
			packet.output_key = dst->pathname + "/" + std::to_string(ingress_binding.task.job_id);
			packet.wire_packet.resize(header_size + payload.size());
			header.to_bytes(reinterpret_cast<uint8_t*>(packet.wire_packet.data()));
			if(!payload.empty()) {
				std::memcpy(
					packet.wire_packet.data() + header_size, payload.data(), payload.size());
			}

			scheduler::DecisionGateKey key{
				.task =
					scheduler::TaskRef{
						ingress_binding.task.graph_id,
						ingress_binding.task.job_id,
						ingress_binding.task.task_id,
					},
				.target_task_id = downstream_id,
			};
			_decision_gate.enqueue(key, std::move(packet));
			const std::size_t flushed = flush_key_locked(key, typed_ctxt);
			spdlog::debug("[Worker:{}][gate] finish enqueue graph={} job={} src={} dst={} flushed={}",
						 _name,
						 key.task.graph_id,
						 key.task.job_id,
						 key.task.task_id,
						 key.target_task_id,
						 flushed);
		}
	}

	void on_scheduler_command(const std::span<const std::byte>& payload,
							  DefaultCascadeContextType* typed_ctxt,
							  uint32_t worker_id) {
		auto perf_timer = make_perf_timer("on_scheduler_command");
		(void)perf_timer;

		if(!typed_ctxt) {
			return;
		}

		auto* buf = reinterpret_cast<const uint8_t*>(payload.data());
		auto cmd = scheduler::SchedulerCommand::from_bytes(nullptr, buf);
		if(!cmd) {
			spdlog::warn("[Worker:{}]: failed to decode scheduler command", _name);
			return;
		}
		log_perf_counter("scheduler_command.decisions", cmd->decisions.size());
		log_perf_counter("scheduler_command.cancellations", cmd->cancellations.size());

		std::lock_guard<std::mutex> lock(_decision_gate_mu);
		for(const auto& decision : cmd->decisions) {
			if(decision.worker_id != static_cast<uint16_t>(worker_id)) {
				spdlog::debug("[Worker:{}][gate] skip decision for worker={} local_worker={} graph={} job={} src={} dst={}",
						 _name,
						 decision.worker_id,
						 worker_id,
						 decision.task.graph_id,
						 decision.task.job_id,
						 decision.task.task_id,
						 decision.target_task_id);
				continue;
			}
			scheduler::DecisionGateKey key{
				.task =
					scheduler::TaskRef{
						decision.task.graph_id,
						decision.task.job_id,
						decision.task.task_id,
					},
				.target_task_id = decision.target_task_id,
			};
			spdlog::debug("[Worker:{}][gate] add_credit worker={} graph={} job={} src={} dst={} amount=1",
						 _name,
						 worker_id,
						 key.task.graph_id,
						 key.task.job_id,
						 key.task.task_id,
						 key.target_task_id);
			_decision_gate.add_credit(key, 1);
			const std::size_t flushed = flush_key_locked(key, typed_ctxt);
			spdlog::debug("[Worker:{}][gate] after_credit_flush worker={} graph={} job={} src={} dst={} flushed={}",
						 _name,
						 worker_id,
						 key.task.graph_id,
						 key.task.job_id,
						 key.task.task_id,
						 key.target_task_id,
						 flushed);
		}

		for(const auto& cancel : cmd->cancellations) {
			_decision_gate.cancel_task(
				scheduler::TaskRef{ cancel.graph_id, cancel.job_id, cancel.task_id });
		}
	}

	std::size_t flush_key_locked(const scheduler::DecisionGateKey& key,
							DefaultCascadeContextType* typed_ctxt) {
		auto perf_timer = make_perf_timer("flush_key_locked");
		(void)perf_timer;

		auto send_packet = [this, typed_ctxt](PendingPacket&& packet) {
			ObjectWithStringKey out_obj;
			out_obj.key = packet.output_key;
			const size_t total_size = packet.wire_packet.size();
			out_obj.blob = Blob(
				[data = std::move(packet.wire_packet)](uint8_t* out,
													   std::size_t) mutable -> std::size_t {
					if(!data.empty()) {
						std::memcpy(out, data.data(), data.size());
					}
					return data.size();
				},
				total_size);

			typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(
				out_obj, 0, 0, true);
		};

		const size_t flushed = _decision_gate.flush_key(key, send_packet);
		log_perf_counter("flush_key_locked.flushed", flushed);
		if(flushed > 0) {
			spdlog::debug("[Worker:{}]: flushed {} gated outputs for job={} src={} dst={}",
						  _name,
						  flushed,
						  key.task.job_id,
						  key.task.task_id,
						  key.target_task_id);
		}

		return flushed;
	}

	void report_completion(const scheduler::TaskBinding& binding,
						   uint32_t worker_id,
						   DefaultCascadeContextType* typed_ctxt) {
		auto perf_timer = make_perf_timer("report_completion");
		(void)perf_timer;

		if(!typed_ctxt) {
			return;
		}

		size_t pending_count = 0;
		{
			std::lock_guard<std::mutex> lock(_decision_gate_mu);
			pending_count = _decision_gate.pending_packet_count();
		}

		scheduler::WorkerStatus status{
			static_cast<uint16_t>(worker_id),
			binding.task.task_id,
			static_cast<uint16_t>(pending_count),
			0,
			std::vector<scheduler::TaskRef>{
				scheduler::TaskRef{
					binding.task.graph_id, binding.task.job_id, binding.task.task_id },
			},
		};

		const size_t payload_size = status.size_estimate();
		ObjectWithStringKey obj;
		obj.key = "/SCHED/status/" + std::to_string(binding.task.job_id) + "/" +
				  std::to_string(binding.task.task_id);
		obj.blob = Blob(
			[status](uint8_t* out, std::size_t) mutable -> std::size_t {
				status.to_bytes(out);
				return status.size_estimate();
			},
			payload_size);

		typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(
			obj, 0, 0, true);
		log_perf_counter("report_completion.pending_count", pending_count);
		spdlog::info("[Worker:{}]: reported completion job={} task={} pending={}",
					 _name,
					 binding.task.job_id,
					 binding.task.task_id,
					 pending_count);
	}

	void start_executor_if_needed() {
		auto perf_timer = make_perf_timer("start_executor_if_needed");
		(void)perf_timer;

		if(_exec_thread.joinable()) {
			return;
		}
		_exec_thread = std::thread([this]() {
			while(true) {
				DefaultCascadeContextType* typed_ctxt = nullptr;
				uint32_t worker_id = 0;
				{
					std::unique_lock<std::mutex> lock(_exec_mu);
					_exec_cv.wait(lock, [this]() { return _exec_stop || _exec_has_work; });
					if(_exec_stop) {
						break;
					}
					typed_ctxt = _last_ctxt;
					worker_id = _last_worker_id;
					_exec_has_work = false;
				}

				if(!_join_service) {
					continue;
				}

				// Currently, there is no free and so memory leaks are expected
				// auto drain_timer = make_perf_timer("executor.drain_ingress");
				// const std::size_t drained = _join_service->drain_ingress(256);
				// log_perf_counter("executor.drain_ingress.count", drained);
				// (void)drain_timer;

				while(auto binding = _join_service->poll_ready()) {
					try {
						auto execute_timer = make_perf_timer("executor.execute_udl");
						execute_udl(*binding, typed_ctxt, worker_id);
						(void)execute_timer;

						auto free_timer = make_perf_timer("executor.free_binding");
						_join_service->free(*binding);
						(void)free_timer;
					} catch(const std::exception& ex) {
						spdlog::error("[Worker:{}]: execute thread exception: {}", _name, ex.what());
					} catch(...) {
						spdlog::error("[Worker:{}]: execute thread unknown exception", _name);
					}
				}

				const std::size_t extra = _join_service->drain_ingress(256);
				log_perf_counter("executor.extra_drain.count", extra);
				if(extra > 0) {
					std::lock_guard<std::mutex> lock(_exec_mu);
					_exec_has_work = true;
					_exec_cv.notify_one();
				}
			}
		});
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
		auto perf_timer = make_perf_timer("observer_callback");
		(void)sender;
		(void)prefix_length;
		(void)perf_timer;

		(void)version;
		(void)outputs;

		std::call_once(_init_flag, [this]() {
			_registry = std::make_unique<scheduler::DagRegistry>(
				scheduler::DagRegistry::from_dfg_file("jobs.json", _uuid));
			_join_service = std::make_unique<scheduler::TaskJoinService>(*_registry);
			initialize_resources();
			start_executor_if_needed();
			spdlog::info("[Worker:{}]: initialized", _name);
		});

		if(!_registry) {
			spdlog::error("[Worker:{}]: registry failed to initialize", _name);
			return;
		}

		const auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
		if(!typed_ctxt) {
			spdlog::error("[Worker:{}]: empty cascade context", _name);
			return;
		}

		const auto obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
		if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
			spdlog::error("[Worker:{}]: empty blob object", _name);
			return;
		}

		// at this point, the blob object either encodes a task output or scheduler command
		// the way to differentiate between to two is to compare path prefixes (_data_path) vs.
		// _scheduler_path

		// NOTE: std::span is a slice type, meaning it is a non-owning, non-writable view into a slice of data.
		// Since the view is a slice within the SST table, we must extend the lifetime via a memcpy if we want
		// to process the data outside the lifetime of this method.

		// the fast path to determine if the message is a data or control message is to just look
		// at the scheduler path and check if the prefixes are equal
		const bool is_control_message =
			key_string.compare(0, _scheduler_path.size(), _scheduler_path) == 0;
		const std::span<const std::byte> payload_slice(
			reinterpret_cast<const std::byte*>(obj->blob.bytes), obj->blob.size);

		if(is_control_message) {
			auto control_timer = make_perf_timer("observer.control_message");
			on_scheduler_command(payload_slice, typed_ctxt, worker_id);
			(void)control_timer;
		} else {
			auto data_timer = make_perf_timer("observer.data_message");
			spdlog::info("[Worker:{}] ingest key={}", _name, key_string);
			if(_join_service->try_ingest(key_string, payload_slice)) {
				log_perf_counter("observer.ingest.accepted", 1u);
				std::lock_guard<std::mutex> lock(_exec_mu);
				_last_ctxt = typed_ctxt;
				_last_worker_id = worker_id;
				_exec_has_work = true;
				_exec_cv.notify_one();
			} else {
				log_perf_counter("observer.ingest.accepted", 0u);
				spdlog::warn("[Worker:{}]: ingress queue full, dropping key={}", _name, key_string);
			}
			(void)data_timer;
		}
	}
};

} // namespace cascade
} // namespace derecho
