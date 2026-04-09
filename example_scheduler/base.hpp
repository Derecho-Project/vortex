#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>
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

private:
	/// @brief name of this udl
	std::string _name;

	/// @brief path in which this udl receives data messages
	std::string _data_path;

	/// @brief path in which this udl receives scheduling messages
	std::string _scheduler_path;

	/// @brief uuid of of this udl that is not locked behind a compile time macro
	std::string _uuid;

	/// @brief initialization guard
	bool _initialized = false;

protected:
	std::unique_ptr<scheduler::DagRegistry> _registry;
	std::unique_ptr<scheduler::TaskJoinService> _join_service;
	scheduler::DecisionGate<PendingPacket> _decision_gate;
	std::mutex _decision_gate_mu;

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
		if(!typed_ctxt || !_registry) {
			return;
		}

		const auto* src = _registry->find_task(ingress_binding.task.graph_id, ingress_binding.task.task_id);
		if(src == nullptr) {
			spdlog::warn("[Worker:{}]: no DAG node for source task {}", _name, ingress_binding.task.task_id);
			return;
		}

		std::lock_guard<std::mutex> lock(_decision_gate_mu);
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
				std::memcpy(packet.wire_packet.data() + header_size, payload.data(), payload.size());
			}

			scheduler::DecisionGateKey key {
				.task = scheduler::TaskRef {
					ingress_binding.task.graph_id,
					ingress_binding.task.job_id,
					ingress_binding.task.task_id,
				},
				.target_task_id = downstream_id,
			};
			_decision_gate.enqueue(key, std::move(packet));
			flush_key_locked(key, typed_ctxt);
		}
	}

	void on_scheduler_command(const std::span<const std::byte>& payload,
						 DefaultCascadeContextType* typed_ctxt,
						 uint32_t worker_id) {
		if(!typed_ctxt) {
			return;
		}

		auto* buf = reinterpret_cast<const uint8_t*>(payload.data());
		auto cmd = scheduler::SchedulerCommand::from_bytes(nullptr, buf);
		if(!cmd) {
			spdlog::warn("[Worker:{}]: failed to decode scheduler command", _name);
			return;
		}

		std::lock_guard<std::mutex> lock(_decision_gate_mu);
		for(const auto& decision : cmd->decisions) {
			if(decision.worker_id != worker_id) {
				continue;
			}
			scheduler::DecisionGateKey key {
				.task = scheduler::TaskRef {
					decision.task.graph_id,
					decision.task.job_id,
					decision.task.task_id,
				},
				.target_task_id = decision.target_task_id,
			};
			_decision_gate.add_credit(key, 1);
			flush_key_locked(key, typed_ctxt);
		}

		for(const auto& cancel : cmd->cancellations) {
			_decision_gate.cancel_task(
				scheduler::TaskRef {cancel.graph_id, cancel.job_id, cancel.task_id});
		}
	}

	void flush_key_locked(const scheduler::DecisionGateKey& key,
					 DefaultCascadeContextType* typed_ctxt) {
		auto send_packet = [this, typed_ctxt](PendingPacket&& packet) {
			ObjectWithStringKey out_obj;
			out_obj.key = packet.output_key;
			const size_t total_size = packet.wire_packet.size();
			out_obj.blob = Blob(
				[data = std::move(packet.wire_packet)](uint8_t* out, std::size_t) mutable -> std::size_t {
					if(!data.empty()) {
						std::memcpy(out, data.data(), data.size());
					}
					return data.size();
				},
				total_size);

			typed_ctxt->get_service_client_ref()
				.put_and_forget<VolatileCascadeStoreWithStringKey>(out_obj, 0, 0, true);
		};

		const size_t flushed = _decision_gate.flush_key(key, send_packet);
		if(flushed > 0) {
			spdlog::debug("[Worker:{}]: flushed {} gated outputs for job={} src={} dst={}",
						 _name,
						 flushed,
						 key.task.job_id,
						 key.task.task_id,
						 key.target_task_id);
		}
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
		(void)version;
		(void)outputs;

		if(!_initialized) {
			_initialized = true;
			_registry = std::make_unique<scheduler::DagRegistry>(
				scheduler::DagRegistry::from_dfg_file("jobs.json", _uuid));
			_join_service = std::make_unique<scheduler::TaskJoinService>(*_registry);
			initialize_resources();
			spdlog::info("[Worker:{}]: initialized", _name);
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
		const std::string_view path_prefix(
			key_string.data(),
			prefix_length <= key_string.size() ? prefix_length : key_string.size());
		auto normalize_path = [](std::string_view path) {
			while(path.size() > 1 && path.back() == '/') {
				path.remove_suffix(1);
			}
			return path;
		};
		const std::string_view normalized_prefix = normalize_path(path_prefix);
		const std::span<const std::byte> payload_slice(
			reinterpret_cast<const std::byte*>(obj->blob.bytes), obj->blob.size);
		if(normalized_prefix == normalize_path(_scheduler_path)) {
			on_scheduler_command(payload_slice, typed_ctxt, worker_id);
		} else {
			auto* buf = reinterpret_cast<const uint8_t*>(payload_slice.data());
			auto header = scheduler::TaskOutput::from_bytes(nullptr, buf);
			if(!header) {
				spdlog::error("[Worker:{}]: non-scheduler message is not TaskOutput: {}", _name, normalized_prefix);
				return;
			}

			const auto* dst_task = _registry->find_task(header->graph_id, header->target_task_id);
			if(dst_task == nullptr) {
				spdlog::error("[Worker:{}]: unknown target task graph={} task={}",
							 _name,
							 header->graph_id,
							 header->target_task_id);
				return;
			}

			if(dst_task->udl_uuid != _uuid) {
				spdlog::debug("[Worker:{}]: task output is for different UDL uuid={}, skipping",
							 _name,
							 dst_task->udl_uuid);
				return;
			}

			if(normalized_prefix != normalize_path(dst_task->pathname)) {
				spdlog::warn("[Worker:{}]: key prefix '{}' does not match registry pathname '{}' for task {}",
							 _name,
							 normalized_prefix,
							 dst_task->pathname,
							 dst_task->task_id);
				return;
			}

			spdlog::info("[Worker:{}] ingest key={}", _name, key_string);
			auto binding = _join_service->recv(key_string, payload_slice);
			if(binding) {
				// safe use of "unsafe" method because we check if it exists in the guard
				execute_udl(*binding, typed_ctxt, worker_id);
				_join_service->free(*binding);
			}
		}
	}
};

} // namespace cascade
} // namespace derecho
