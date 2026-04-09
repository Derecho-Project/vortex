#include <vortex_scheduler/task_join_service.hpp>

VORTEX_SCHEDULER_NAMESPACE_BEGIN

TaskJoinService::TaskJoinService(const DagRegistry& dag_registry)
	: _dag_registry(dag_registry) { }

std::optional<TaskBinding> TaskJoinService::recv(const std::string_view& string_key,
												 const std::span<const std::byte>& bytes) {
	if(bytes.empty()) {
		return std::nullopt;
	}

	const auto* buf = reinterpret_cast<const uint8_t*>(bytes.data());
	auto header = TaskOutput::from_bytes(nullptr, buf);
	if(!header) {
		return std::nullopt;
	}

	const auto header_size = header->size_estimate();
	if(bytes.size() < header_size || bytes.size() < header_size + header->payload_size) {
		return std::nullopt;
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

	const auto* node = _dag_registry.find_task(header->graph_id, header->target_task_id);
	if(node == nullptr) {
		return std::nullopt;
	}

	const auto* upstream = _dag_registry.find_upstream(header->graph_id, header->target_task_id);
	const uint16_t expected_inputs = upstream ? static_cast<uint16_t>(upstream->size()) : 0;
	uint16_t dependency_slot = 0;
	if(upstream) {
		for(std::size_t i = 0; i < upstream->size(); ++i) {
			if(upstream->at(i) == header->source_task_id) {
				dependency_slot = static_cast<uint16_t>(i);
				break;
			}
		}
	}

	TaskRef task_ref{header->graph_id, header->job_id, header->target_task_id};
	const uint16_t processor_id = header->target_task_id;

	return _join_table.add_input(task_ref,
								 processor_id,
								 expected_inputs,
								 dependency_slot,
								 bh);
}

std::optional<std::span<const std::byte>> TaskJoinService::resolve(const BlobHandle& handle) const {
	auto it = _payload_handles.find(handle.segment_id);
	if(it == _payload_handles.end()) {
		return std::nullopt;
	}

	return _payload_arena.resolve_const(it->second);
}

void TaskJoinService::free(const TaskBinding& binding) {
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
