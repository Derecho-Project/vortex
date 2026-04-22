#include <optional>
#include <memory_resource>
#include <vortex_scheduler/udl_base.hpp>

using namespace derecho::cascade;
namespace scheduler {

namespace {

/// @brief view a blob as a byte span
std::span<const std::byte> as_byte_span(const Blob& blob) {
	return std::span<const std::byte>(reinterpret_cast<const std::byte*>(blob.bytes), blob.size);
}
}; // namespace

VortexWorkerUdl::VortexWorkerUdl(const std::string_view& data_path,
								 const std::string_view& scheduler_path,
								 const std::string_view& uuid,
								 const std::filesystem::path& dfg_path,
								 const std::filesystem::path& timestamp_log_path,
								 const std::size_t arena_segment_capacity)
	: _data_path(data_path)
	, _scheduler_path(scheduler_path)
	, _uuid(uuid)
	, _logpath(timestamp_log_path == "" ? std::filesystem::path("timestamps/" + std::string(uuid) + ".log")
										: timestamp_log_path)
	, _registry(dfg_path, uuid)
	, _ocdpo_arena(arena_segment_capacity)
	, _ingress_queue()
	, _ingress_ptok(_ingress_queue)
	, _ingress_ctok(_ingress_queue)
	, _execution_queue()
	, _execution_ptok(_execution_queue)
	, _execution_ctok(_execution_queue)
	, _notify_finish_queue()
	, _notify_finish_ptok(_notify_finish_queue)
	, _notify_finish_ctok(_notify_finish_queue)
	, _cleanup_queue()
	, _cleanup_ptok(_cleanup_queue)
	, _cleanup_ctok(_cleanup_queue)
	, _decision_queue()
	, _decision_ptok(_decision_queue)
	, _decision_ctok(_decision_queue)
	, _stop(false)
	, _ingress_thread(std::thread(&VortexWorkerUdl::ingress_loop, this))
	, _execution_thread(std::thread(&VortexWorkerUdl::execution_loop, this))
	, _notify_finish_thread(std::thread(&VortexWorkerUdl::notify_finish_loop, this))
	, _timestamp_flush_thread(std::thread(&VortexWorkerUdl::timestamp_flush_loop, this)) { }

VortexWorkerUdl::~VortexWorkerUdl() {
	_stop.store(true, std::memory_order_release);
	if(_ingress_thread.joinable()) {
		_ingress_thread.join();
	}
	if(_execution_thread.joinable()) {
		_execution_thread.join();
	}
	if(_notify_finish_thread.joinable()) {
		_notify_finish_thread.join();
	}
	if(_timestamp_flush_thread.joinable()) {
		_timestamp_flush_thread.join();
	}
}

void VortexWorkerUdl::operator()(const derecho::node_id_t sender,
								 const std::string& key_string,
								 const uint32_t prefix_length,
								 persistent::version_t version,
								 const mutils::ByteRepresentable* const value_ptr,
								 const std::unordered_map<std::string, bool>& outputs,
								 ICascadeContext* ctxt,
								 uint32_t worker_id) {

	std::call_once(_init_flag, [this, ctxt, worker_id]() {
		_cascade_context = dynamic_cast<DefaultCascadeContextType*>(ctxt);
		_node_id = worker_id;
		std::filesystem::create_directories(_logpath.parent_path());
		initialize_resources();
	});

	const auto obj = dynamic_cast<const ObjectWithStringKey*>(value_ptr);
	if(obj == nullptr || obj->blob.bytes == nullptr || obj->blob.size == 0) {
		return;
	}

	// at this point, the blob object either encodes a task output or scheduler command
	// the way to differentiate between to two is to compare path prefixes (_data_path) vs.
	// _scheduler_path

	// NOTE: std::span is a slice type, meaning it is a non-owning, non-writable view into a slice of data.
	// Since the view is a slice within the SST table, we must extend the lifetime via a memcpy if we want
	// to process the data outside the lifetime of this method.

	// zero malloc path after initial queue warmup
	const bool is_control_message = key_string.compare(0, _scheduler_path.size(), _scheduler_path) == 0;
	if(is_control_message) {
		auto scheduler_message = scheduler::message::SchedulerCommandView::from_buffer(as_byte_span(obj->blob));
		for(const auto& decision : scheduler_message.decisions) {
			_decision_queue.enqueue(_decision_ptok, decision);
		}
		if(!scheduler_message.cancellations.empty()) {
			spdlog::warn("Received cancellations; unsupported");
		}
	} else {
		// std::cout << "data message received: " << key_string << std::endl;
		auto ingress_time = std::chrono::steady_clock::now().time_since_epoch().count();
		auto bytes = as_byte_span(obj->blob);
		auto task_message = scheduler::message::TaskOutputView::from_buffer(bytes);
		timestamp::DyeLogger::log_ingress(_node_id, task_message.decision.target_task);

		auto blob_handle = _ocdpo_arena.put(task_message.payload);
		auto decision = task_message.decision;
		_ingress_queue.enqueue(_ingress_ptok,
							   IngressMessage{
								   .decision = decision,
								   .payload_handle = blob_handle,
							   });
	}

	// clean up arena via _cleanup_queue
	while(!_cleanup_queue.size_approx() == 0) {
		std::array<TaskBinding, 64> batch;
		std::size_t tasks_popped = _cleanup_queue.try_dequeue_bulk(_cleanup_ctok, batch.data(), batch.size());
		for(std::size_t i = 0; i < tasks_popped; ++i) {
			for(std::size_t j = 0; j < batch[i].num_upstream; ++j) {
				_ocdpo_arena.take(std::move(batch[i].payload_handles[j]));
			}
		}
	}
}

void VortexWorkerUdl::ingress_loop() {
	constexpr double LOAD_FACTOR = 0.7;

	constexpr std::size_t BATCH_SIZE = 64;
	constexpr std::size_t UNORDERED_MAP_SIZE = 128;
	constexpr std::size_t INITIAL_SIZE = static_cast<std::size_t>(UNORDERED_MAP_SIZE / LOAD_FACTOR) + 1;

	std::pmr::unsynchronized_pool_resource pool;
	std::pmr::unordered_map<std::uint64_t, std::pmr::vector<IngressMessage>> join_table{ INITIAL_SIZE, &pool };
	join_table.max_load_factor(LOAD_FACTOR);

	auto process_message = [&](IngressMessage&& message) {
		auto dependencies =
			_registry.get_upstream_tasks(message.decision.target_task.graph_id, message.decision.target_task.task_id);
		auto id = static_cast<std::size_t>(message.decision.target_task.graph_id) << 48 |
				  static_cast<std::size_t>(message.decision.target_task.job_id) << 32 |
				  static_cast<std::size_t>(message.decision.target_task.task_id) << 16 |
				  static_cast<std::size_t>(message.decision.target_task.stream_id);

		// fast path for if there are no dependencies
		if(dependencies.empty()) {
			return std::make_optional(std::move(TaskBinding{
				.upstream = { },
				.payload_handles = { message.payload_handle },
				.current_task = message.decision.target_task,
				.num_upstream = 0,
			}));
		}

		auto& array = join_table[id];
		array.emplace_back(std::move(message));

		if(array.size() == dependencies.size()) {
			// all dependencies have arrived, we can dispatch the task for execution

			TaskBinding binding{
				.upstream = { },
				.payload_handles = { },
				.current_task = array.front().decision.target_task,
				.num_upstream = static_cast<std::uint8_t>(array.size()),
			};

			for(std::size_t dependency_index = 0; dependency_index < dependencies.size(); ++dependency_index) {
				const auto dependency_task_id = dependencies[dependency_index];
				for(const auto& ingress : array) {
					if(ingress.decision.source_task.task_id == dependency_task_id) {
						binding.upstream[dependency_index] = ingress.decision.source_task;
						binding.payload_handles[dependency_index] = ingress.payload_handle;
						break;
					}
				}
			}

			join_table.erase(id);
			return std::make_optional(std::move(binding));
		}

		return std::optional<TaskBinding>{ };
	};

	std::array<IngressMessage, BATCH_SIZE> batch;
	std::array<TaskBinding, BATCH_SIZE> ready_tasks;
	while(!_stop.load(std::memory_order_acquire)) {
		std::size_t messages_popped = _ingress_queue.try_dequeue_bulk(_ingress_ctok, batch.data(), batch.size());

		if(join_table.size() + batch.size() > UNORDERED_MAP_SIZE) {
			spdlog::warn(
				"Join table size {} is approaching the threshold of {}", join_table.size(), UNORDERED_MAP_SIZE);
		}

		if(messages_popped == 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(5));
			continue;
		}

		std::size_t j = 0;
		for(std::size_t i = 0; i < messages_popped; ++i) {
			auto result = process_message(std::move(batch[i]));
			if(result.has_value()) {
				timestamp::DyeLogger::log_join(_node_id, result.value().current_task);
				ready_tasks[j++] = std::move(result.value());
			}
		}
		_execution_queue.enqueue_bulk(_execution_ptok, ready_tasks.data(), j);
	}
}

void VortexWorkerUdl::execution_loop() {
	std::array<TaskBinding, 64> batch;
#ifndef NDEBUG
	std::unordered_map<std::uint64_t, std::uint64_t> stream_guard;
#endif
	while(!_stop.load(std::memory_order_acquire)) {
		std::size_t tasks_popped = _execution_queue.try_dequeue_bulk(_execution_ctok, batch.data(), batch.size());
		if(tasks_popped == 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(5));
			continue;
		}

		for(std::size_t i = 0; i < tasks_popped; ++i) {
#ifndef NDEBUG
			// TODO: implement me: assert that job ids within a stream are monotonically increasing
#endif
			timestamp::DyeLogger::log_execution(_node_id, batch[i].current_task);
			execute_udl(std::move(batch[i]));
		}
	}
}

void VortexWorkerUdl::notify_finish_loop() {
	constexpr std::size_t BATCH_SIZE = 64;
	std::array<TaskBinding, BATCH_SIZE> batch;

	while(!_stop.load(std::memory_order_acquire)) {
		const std::size_t popped =
			_notify_finish_queue.try_dequeue_bulk(_notify_finish_ctok, batch.data(), batch.size());

		if(popped == 0) {
			std::this_thread::sleep_for(std::chrono::microseconds(5));
			continue;
		}

		// Aggregate all completed tasks in this batch into a single
		// WorkerStatus message addressed to the scheduler (subgroup 0,
		// shard 0). The worker identifies itself by its cascade node id.
		const auto worker_id = static_cast<std::uint16_t>(_cascade_context->get_service_client_ref().get_my_id());

		std::vector<message::TaskRef> completed;
		completed.reserve(popped);
		for(std::size_t i = 0; i < popped; ++i) {
			completed.push_back(batch[i].current_task);
		}

		message::WorkerStatus status{
			.worker = message::WorkerRef{ .shard_number = worker_id },
			.task_id = 0,
			.queue_size = _execution_queue.size_approx(),
			.empty_ns = 0,
			.completed = std::move(completed),
		};

		const std::size_t total_size = status.size_estimate();

		auto blob = Blob(
			[s = std::move(status)](std::uint8_t* out, std::size_t cap) -> std::size_t {
				const auto out_span = std::span<std::byte>(reinterpret_cast<std::byte*>(out), cap);
				return s.to_buffer(out_span);
			},
			total_size);

		// Everything to the scheduler is published under /SCHED

		ObjectWithStringKey obj("/SCHED/notif", std::move(blob));

		try {
			_cascade_context->get_service_client_ref().template put_and_forget<VolatileCascadeStoreWithStringKey>(
				obj, /*subgroup_index*/ 0, /*shard_index*/ 0, true);
		} catch(const std::exception& e) {
			spdlog::error("notify_finish_loop: failed to send WorkerStatus: {}", e.what());
		}
	}
}

void VortexWorkerUdl::timestamp_flush_loop() {
	while(!_stop.load(std::memory_order_acquire)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
		timestamp::DyeLogger::flush(_logpath);
	}
}

std::span<const std::byte> VortexWorkerUdl::view(const Arena::BufferHandle& handle) const noexcept {
	return _ocdpo_arena.view(handle);
}

void VortexWorkerUdl::notify_finish(const TaskBinding& binding) {
#ifndef NDEBUG
	// Per-instance thread affinity: notify_finish must always be called from
	// the same thread to preserve linearizability of completion notifications.
	// (Function-level statics would be shared across all worker DLL instances
	// in this process since they live in libvortex_scheduler.so.)
	std::call_once(_notify_finish_tid_flag, [this]() { _notify_finish_tid = std::this_thread::get_id(); });

	assert(_notify_finish_tid == std::this_thread::get_id() &&
		   "notify_finish must be called from the same thread to guarantee "
		   "linearizability of task completion notifications");
#endif
	_notify_finish_queue.enqueue(_notify_finish_ptok, binding);
}

}; // namespace scheduler
