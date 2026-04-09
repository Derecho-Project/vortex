#include <vortex_scheduler/task_join_service.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

TaskJoinService::TaskJoinService(const DagRegistry& dag_registry)
	: _dag_registry(dag_registry) {
	constexpr std::size_t default_capacity = 1024;
	const std::size_t capacity = next_pow2(default_capacity);
	_ingress_ring = std::make_unique<IngressSlot[]>(capacity);
	_ingress_capacity = capacity;
	for(std::size_t i = 0; i < capacity; ++i) {
		_ingress_ring[i].seq.store(i, std::memory_order_relaxed);
	}
	_ingress_mask = capacity - 1;
}

std::size_t TaskJoinService::next_pow2(std::size_t value) {
	if(value <= 1) {
		return 1;
	}
	--value;
	for(std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
		value |= value >> shift;
	}
	return value + 1;
}

bool TaskJoinService::claim_owner_thread() {
	const std::size_t token = std::hash<std::thread::id>{}(std::this_thread::get_id());
	if(token == 0) {
		return false;
	}

	std::size_t expected = 0;
	if(_owner_thread_token.compare_exchange_strong(expected,
													   token,
													   std::memory_order_acq_rel,
													   std::memory_order_acquire)) {
		return true;
	}

	return _owner_thread_token.load(std::memory_order_acquire) == token;
}

bool TaskJoinService::try_enqueue_ingress(IngressPacket&& packet) {
	if(_ingress_capacity == 0) {
		return false;
	}

	std::size_t head = _ingress_head.load(std::memory_order_relaxed);
	for(;;) {
		IngressSlot& slot = _ingress_ring[head & _ingress_mask];
		const std::size_t seq = slot.seq.load(std::memory_order_acquire);
		const std::intptr_t diff = static_cast<std::intptr_t>(seq) -
									 static_cast<std::intptr_t>(head);
		if(diff == 0) {
			if(_ingress_head.compare_exchange_weak(head,
												  head + 1,
												  std::memory_order_acq_rel,
												  std::memory_order_relaxed)) {
				slot.packet = std::move(packet);
				slot.seq.store(head + 1, std::memory_order_release);
				return true;
			}
			continue;
		}
		if(diff < 0) {
			return false;
		}

		head = _ingress_head.load(std::memory_order_relaxed);
	}
}

bool TaskJoinService::try_dequeue_ingress(IngressPacket& packet) {
	if(_ingress_capacity == 0) {
		return false;
	}

	std::size_t tail = _ingress_tail.load(std::memory_order_relaxed);
	for(;;) {
		IngressSlot& slot = _ingress_ring[tail & _ingress_mask];
		const std::size_t seq = slot.seq.load(std::memory_order_acquire);
		const std::intptr_t diff = static_cast<std::intptr_t>(seq) -
									 static_cast<std::intptr_t>(tail + 1);
		if(diff == 0) {
			_ingress_tail.store(tail + 1, std::memory_order_relaxed);
			packet = std::move(slot.packet);
			slot.seq.store(tail + _ingress_mask + 1, std::memory_order_release);
			return true;
		}
		if(diff < 0) {
			return false;
		}

		tail = _ingress_tail.load(std::memory_order_relaxed);
	}
}

bool TaskJoinService::try_ingest(const std::string_view& string_key,
								 const std::span<const std::byte>& bytes) {
	if(bytes.empty()) {
		return false;
	}

	IngressPacket packet;
	packet.key.assign(string_key.begin(), string_key.end());
	packet.bytes.assign(bytes.begin(), bytes.end());
	return try_enqueue_ingress(std::move(packet));
}

std::size_t TaskJoinService::drain_ingress(std::size_t max_messages) {
	if(!claim_owner_thread()) {
		return 0;
	}

	std::size_t drained = 0;
	IngressPacket packet;
	while(drained < max_messages && try_dequeue_ingress(packet)) {
		const std::span<const std::byte> bytes(packet.bytes.data(), packet.bytes.size());
		auto binding = process_packet(bytes);
		if(binding) {
			_ready_bindings.emplace_back(std::move(*binding));
		}
		packet.key.clear();
		packet.bytes.clear();
		++drained;
	}

	return drained;
}

std::optional<TaskBinding> TaskJoinService::poll_ready() {
	if(!claim_owner_thread()) {
		return std::nullopt;
	}

	if(_ready_bindings.empty()) {
		return std::nullopt;
	}

	TaskBinding binding = std::move(_ready_bindings.front());
	_ready_bindings.pop_front();
	return binding;
}

std::optional<TaskBinding> TaskJoinService::recv(const std::string_view& string_key,
												 const std::span<const std::byte>& bytes) {
	if(!try_ingest(string_key, bytes)) {
		return std::nullopt;
	}
	(void)drain_ingress(1);
	return poll_ready();
}


std::optional<TaskBinding> TaskJoinService::process_packet(const std::span<const std::byte>& bytes) {
	const auto* buf = reinterpret_cast<const uint8_t*>(bytes.data());
	auto header = TaskOutput::from_bytes_noalloc_const(nullptr, buf);
	if(!header) {
		return std::nullopt;
	}

	const auto header_size = header->size_estimate();
	if(bytes.size() < header_size || bytes.size() < header_size + header->payload_size) {
		return std::nullopt;
	}

	const auto* node = _dag_registry.find_task(header->graph_id, header->target_task_id);
	if(node == nullptr) {
		return std::nullopt;
	}

	const auto* upstream = _dag_registry.find_upstream(header->graph_id, header->target_task_id);
	uint16_t expected_inputs = 1;
	uint16_t dependency_slot = 0;
	if(upstream && !upstream->empty()) {
		expected_inputs = static_cast<uint16_t>(upstream->size());
		bool found_slot = false;
		for(std::size_t i = 0; i < upstream->size(); ++i) {
			if(upstream->at(i) == header->source_task_id) {
				dependency_slot = static_cast<uint16_t>(i);
				found_slot = true;
				break;
			}
		}
		if(!found_slot) {
			return std::nullopt;
		}
	}

	const auto* payload_begin = bytes.data() + header_size;
	std::span<const std::byte> payload(payload_begin, header->payload_size);

	const uint64_t payload_id = _next_payload_id++;
	const BufferHandle arena_handle = _payload_arena.put(payload_id, payload);
	_payload_handles.emplace(payload_id, arena_handle);

	BlobHandle bh;
	bh.pool_class = 0;
	bh.segment_id = static_cast<uint32_t>(payload_id);
	bh.offset = static_cast<uint32_t>(arena_handle.offset);
	bh.size = static_cast<uint32_t>(arena_handle.len);

	TaskRef task_ref{header->graph_id, header->job_id, header->target_task_id};
	const uint16_t processor_id = header->target_task_id;

	return _join_table.add_input(task_ref,
								 processor_id,
								 expected_inputs,
								 dependency_slot,
								 bh);
}

std::optional<std::span<const std::byte>> TaskJoinService::resolve(const BlobHandle& handle) const {
	if(!const_cast<TaskJoinService*>(this)->claim_owner_thread()) {
		return std::nullopt;
	}

	auto it = _payload_handles.find(handle.segment_id);
	if(it == _payload_handles.end()) {
		return std::nullopt;
	}

	return _payload_arena.resolve_const(it->second);
}

void TaskJoinService::free(const TaskBinding& binding) {
	if(!claim_owner_thread()) {
		return;
	}

	for(const auto& input : binding.inputs) {
		auto it = _payload_handles.find(input.segment_id);
		if(it == _payload_handles.end()) {
			continue;
		}
		_payload_arena.take(it->first);
		_payload_handles.erase(it);
	}
}

VORTEX_SCHEDULER_NAMESPACE_END
