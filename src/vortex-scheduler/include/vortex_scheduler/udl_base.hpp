#include <array>
#include <atomic>
#include <chrono>
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
#include <condition_variable>
#include <spdlog/logger.h>
#include <unordered_map>

#include <cascade/object.hpp>
#include <cascade/service_client_api.hpp>
#include <cascade/user_defined_logic_interface.hpp>

#include <concurrentqueue.h>

#include "arena.hpp"
#include "dag_registry.hpp"
#include "messages.hpp"
#include "serde.hpp"
#include "dye.hpp"

// Generates the standard UDL DLL entrypoints for an OCDPO type.
// The OCDPO type must provide static initialize() and get() methods,
// and declare `static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;`.
#define VORTEX_DEFINE_UDL_ENTRYPOINTS(OCDPOType)                                                                       \
	std::shared_ptr<derecho::cascade::OffCriticalDataPathObserver> OCDPOType::ocdpo_ptr;                               \
	void initialize(derecho::cascade::ICascadeContext* ctxt) {                                                         \
		(void)ctxt;                                                                                                    \
		OCDPOType::initialize();                                                                                       \
	}                                                                                                                  \
	std::shared_ptr<derecho::cascade::OffCriticalDataPathObserver> get_observer(                                       \
		derecho::cascade::ICascadeContext* ctxt, const nlohmann::json& cfg) {                                          \
		(void)ctxt;                                                                                                    \
		(void)cfg;                                                                                                     \
		return OCDPOType::get();                                                                                       \
	}                                                                                                                  \
	void release(derecho::cascade::ICascadeContext* ctxt) {                                                            \
		(void)ctxt;                                                                                                    \
	}

// Generates UUID/description exports from MY_UUID and MY_DESC.
// Define both MY_UUID and MY_DESC in the UDL .cpp before invoking this macro.
#define VORTEX_DEFINE_UDL_METADATA(uuid, desc)                                                                         \
	std::string get_uuid() {                                                                                           \
		return uuid;                                                                                                   \
	}                                                                                                                  \
	std::string get_description() {                                                                                    \
		return desc;                                                                                                   \
	}

#define VORTEX_DEFINE_CLASS_METHODS(OCDPOType)                                                                         \
private:                                                                                                               \
	static std::shared_ptr<derecho::cascade::OffCriticalDataPathObserver> ocdpo_ptr;                                   \
                                                                                                                       \
public:                                                                                                                \
	static void initialize() {                                                                                         \
		if(!ocdpo_ptr) {                                                                                               \
			ocdpo_ptr = std::make_shared<OCDPOType>();                                                                 \
		}                                                                                                              \
	}                                                                                                                  \
                                                                                                                       \
	static std::shared_ptr<derecho::cascade::OffCriticalDataPathObserver> get() {                                      \
		return ocdpo_ptr;                                                                                              \
	}

namespace scheduler {

/// @brief wraps common worker logic for cleaner client code.
class VortexWorkerUdl : public derecho::cascade::OffCriticalDataPathObserver {
public:
	VortexWorkerUdl() = delete;
	VortexWorkerUdl(const std::string_view& data_path,
					const std::string_view& scheduler_path,
					const std::string_view& uuid,
					const std::filesystem::path& dfg_path = "jobs.json",
					const std::filesystem::path& timestamp_log_path = "",
					const std::size_t arena_segment_capacity = 1024 * 1024 /* 1 MiB */
	);
	virtual ~VortexWorkerUdl();

public:
	void operator()(const derecho::node_id_t sender,
					const std::string& key_string,
					const uint32_t prefix_length,
					persistent::version_t version,
					const mutils::ByteRepresentable* const value_ptr,
					const std::unordered_map<std::string, bool>& outputs,
					derecho::cascade::ICascadeContext* ctxt,
					uint32_t worker_id) override;

protected:
	/// @brief constructed by the ocdpo handler, consumed by ingress thread
	struct IngressMessage {
		message::Decision decision;
		Arena::BufferHandle payload_handle;
	};

	/// @brief constructed by the ingress thread, consumed by the execution thread
	struct TaskBinding {
		constexpr static std::size_t MAX_SOURCES = 2;
		std::array<message::TaskRef, MAX_SOURCES> upstream;
		std::array<Arena::BufferHandle, MAX_SOURCES> payload_handles;
		message::TaskRef current_task;
		std::uint8_t num_upstream;
	};

	/// @brief
	struct EgressMessage {
		TaskBinding binding;
		derecho::cascade::Blob blob;
	};

	std::string _data_path;
	std::string _scheduler_path;
	std::string _uuid;
	std::filesystem::path _logpath;
	std::uint64_t _node_id = 0;

	DagRegistry _registry;
	Arena _ocdpo_arena;

	// ~~~ incoming messages with consumer thread A ~~~
	moodycamel::ConcurrentQueue<IngressMessage> _ingress_queue;
	moodycamel::ProducerToken _ingress_ptok;
	moodycamel::ConsumerToken _ingress_ctok;

	// ~~~ execution manager with consumer thread B ~~~
	moodycamel::ConcurrentQueue<TaskBinding> _execution_queue;
	moodycamel::ProducerToken _execution_ptok;
	moodycamel::ConsumerToken _execution_ctok;

	// ~~~ send messages to the scheduler with thread C ~~~
	moodycamel::ConcurrentQueue<TaskBinding> _notify_finish_queue;
	moodycamel::ProducerToken _notify_finish_ptok;
	moodycamel::ConsumerToken _notify_finish_ctok;

	// ~~~ remove task bindings and free arena memory with ocdpo thread (which performed the initial memcpy) ~~~
	moodycamel::ConcurrentQueue<TaskBinding> _cleanup_queue;
	moodycamel::ProducerToken _cleanup_ptok;
	moodycamel::ConsumerToken _cleanup_ctok;

	moodycamel::ConcurrentQueue<message::Decision> _decision_queue;
	moodycamel::ProducerToken _decision_ptok;
	moodycamel::ConsumerToken _decision_ctok;

	std::atomic<bool> _stop;
	std::thread _ingress_thread;
	std::thread _execution_thread;
	std::thread _notify_finish_thread;
	std::thread _timestamp_flush_thread;

	derecho::cascade::DefaultCascadeContextType* _cascade_context;

	/// @brief "deferences" a buffer handle to an immutable byte view. UB if handle is stale.
	[[nodiscard]] std::span<const std::byte> view(const Arena::BufferHandle& handle) const noexcept;

protected:
	/// @brief called once the first time UDL receives data. purposes is to lazily load resource intensive computation units.
	virtual void initialize_resources() = 0;

	/// @brief dispatched for a task when upstream dependencies are satisfied
	virtual void execute_udl(TaskBinding&& binding) = 0;

	/// @brief report task completion
	void notify_finish(const TaskBinding& binding);

	/// @brief emit a serializable component which is immediately written into a blob object
	/// allowing for a zero copy path.
	template <serde::BufferSerializable T>
	void emit(TaskBinding&& binding, T&& result) {
#ifndef NDEBUG
		// Per-instance thread affinity. (Function-local statics in template
		// instantiations have vague linkage and get merged across DLLs by the
		// dynamic linker, so per-instance state is required.)
		std::call_once(_emit_tid_flag,
					   [this]() { _emit_tid = std::this_thread::get_id(); });

		assert(_emit_tid == std::this_thread::get_id()
			   && "emit must be called from the same thread to guarantee "
				  "linearizability of task completion notifications");
#endif

		// One staging queue per (T, DLL). The user's invariant is that each
		// vortex worker DLL emits exactly one T, so this is fine -- one
		// dedicated egress thread per worker.
		struct Staged {
			TaskBinding binding;
			T           result;
		};

		static std::once_flag thread_flag;
		static std::thread emit_thread;
		static moodycamel::ConcurrentQueue<Staged> staging_queue;
		static moodycamel::ProducerToken staging_ptok(staging_queue);
		static moodycamel::ConsumerToken staging_ctok(staging_queue);

		std::call_once(thread_flag, [this]() {
			emit_thread = std::thread([this]() {
				// Two caches keyed by `(graph_id, job_id, target_task_id, stream_id)`:
				//  - routing_cache: result is here, waiting for the scheduler's Decision
				//  - decision_cache: Decision is here, waiting for the upstream result
				// Whichever side arrives second triggers a dispatch and evicts the entry.
				std::unordered_map<std::uint64_t, T> routing_cache;
				std::unordered_map<std::uint64_t, message::Decision> decision_cache;

				std::array<message::Decision, 64> decision_batch;
				std::array<Staged, 64> staged_batch;

				auto task_route_to_key = [](const message::TaskRef& ref) {
					return static_cast<std::uint64_t>(ref.graph_id) << 48 |
						   static_cast<std::uint64_t>(ref.job_id) << 32 |
						   static_cast<std::uint64_t>(ref.task_id) << 16 |
						   static_cast<std::uint64_t>(ref.stream_id);
				};

				// Builds a Blob whose generator writes the TaskOutput wire
				// format (matching what the receiver in udl_base.cpp's
				// `operator()` decodes via TaskOutputView::from_buffer):
				//
				//   [MessageComponent::TASK_OUTPUT][Decision]
				//   [size_t payload_size][payload bytes]
				//
				// `result` is move-captured into the lambda; serialization is
				// deferred until Cascade asks for the bytes (true zero-copy:
				// one pass directly into the SST buffer).
				auto build_blob = [](const message::Decision& decision, T result) {
					serde::Sizer sizer;
					sizer.put<message::detail::MessageComponent>();
					sizer.put<message::Decision>();
					sizer.put<std::size_t>(); // payload size
					const std::size_t header_size = sizer.bytes_written();
					const std::size_t total_size = header_size + result.size_estimate();

					return derecho::cascade::Blob(
						[d = decision, r = std::move(result)](std::uint8_t* out,
															  std::size_t cap) -> std::size_t {
							const auto out_span = std::span<std::byte>(
								reinterpret_cast<std::byte*>(out), cap);
							serde::Writer writer{ out_span };
							writer.put(message::detail::MessageComponent::TASK_OUTPUT);
							writer.put(d);
							writer.put<std::size_t>(r.size_estimate());
							const auto payload_written = r.to_buffer(
								out_span.subspan(writer.bytes_written()));
							return writer.bytes_written() + payload_written;
						},
						total_size);
				};

				auto dispatch = [this, &build_blob](const message::Decision& decision, T result) {
					auto blob = build_blob(decision, std::move(result));

					// Resolve the destination pathname from the registry. The
					// receiving worker subscribes on its data_path (e.g. "/B"),
					// so the key prefix must match that path.
					const auto* target_node = _registry.find_node(decision.target_task.task_id);
					if(target_node == nullptr) {
						spdlog::error("emit/dispatch: target task_id {} not in registry; dropping",
									  decision.target_task.task_id);
						return;
					}

					const std::string key = target_node->pathname + "/"
						+ std::to_string(decision.target_task.graph_id) + "/"
						+ std::to_string(decision.target_task.job_id) + "/"
						+ std::to_string(decision.target_task.stream_id);

					derecho::cascade::ObjectWithStringKey obj(key, std::move(blob));

					try {
						timestamp::DyeLogger::log_egress(_node_id, decision.source_task);
						_cascade_context->get_service_client_ref()
							.template put_and_forget<derecho::cascade::VolatileCascadeStoreWithStringKey>(
								obj,
								/*subgroup_index*/ 0,
								/*shard_index*/ decision.target_worker.shard_number,
								true);
					} catch(const std::exception& e) {
						spdlog::error("emit/dispatch: put_and_forget to '{}' failed: {}",
									  key, e.what());
					}
				};

				while(!_stop.load(std::memory_order_acquire)) {
					// ~~~ drain scheduler decisions ~~~
					const std::size_t decisions_popped = _decision_queue.try_dequeue_bulk(
						_decision_ctok, decision_batch.data(), decision_batch.size());

					for(std::size_t i = 0; i < decisions_popped; ++i) {
						auto& decision = decision_batch[i];
						const auto key = task_route_to_key(decision.target_task);
						if(auto it = routing_cache.find(key); it != routing_cache.end()) {
							dispatch(decision, std::move(it->second));
							routing_cache.erase(it);
						} else {
							decision_cache.emplace(key, std::move(decision));
						}
					}

					// ~~~ drain emitted results ~~~
					const std::size_t staged_popped = staging_queue.try_dequeue_bulk(
						staging_ctok, staged_batch.data(), staged_batch.size());

					for(std::size_t i = 0; i < staged_popped; ++i) {
						auto& staged = staged_batch[i];
						const auto& src = staged.binding.current_task;

						// Each downstream task in the DAG is a separate
						// rendezvous target; the same result fans out to all
						// of them.
						const auto downstream = _registry.get_downstream_tasks(
							src.graph_id, src.task_id);

						if(downstream.empty()) {
							continue; // sink task -- nothing to route
						}

						for(std::size_t k = 0; k < downstream.size(); ++k) {
							const message::TaskRef target_ref{
								.graph_id  = src.graph_id,
								.job_id    = src.job_id,
								.task_id   = downstream[k],
								.stream_id = src.stream_id,
							};
							const auto key = task_route_to_key(target_ref);

							// Last fan-out gets the move; earlier ones must copy.
							const bool is_last = (k + 1 == downstream.size());

							if(auto it = decision_cache.find(key); it != decision_cache.end()) {
								if(is_last) {
									dispatch(it->second, std::move(staged.result));
								} else {
									dispatch(it->second, staged.result);
								}
								decision_cache.erase(it);
							} else {
								if(is_last) {
									routing_cache.emplace(key, std::move(staged.result));
								} else {
									routing_cache.emplace(key, staged.result);
								}
							}
						}
					}

					if(decisions_popped == 0 && staged_popped == 0) {
						std::this_thread::sleep_for(std::chrono::microseconds(5));
					}
				}
			});
		});

		// Producer side: hand off to the egress thread.
		timestamp::DyeLogger::log_holding(_node_id, binding.current_task);
		staging_queue.enqueue(staging_ptok,
		                      Staged{ std::move(binding), std::forward<T>(result) });
	}

private:
	std::once_flag _init_flag;
#ifndef NDEBUG
	std::once_flag _notify_finish_tid_flag;
	std::thread::id _notify_finish_tid;
	std::once_flag _emit_tid_flag;
	std::thread::id _emit_tid;
#endif
	void ingress_loop();
	void execution_loop();
	void notify_finish_loop();
	void timestamp_flush_loop();
};

} // namespace scheduler
