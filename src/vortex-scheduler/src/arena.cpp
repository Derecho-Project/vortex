
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <vortex_scheduler/arena.hpp>

namespace scheduler {

Arena::Arena(std::size_t segment_capacity)
	: segment_capacity_(segment_capacity) { }

Arena::BufferHandle Arena::put(const std::span<const std::byte>& value) {
	if (value.size_bytes() >= static_cast<std::size_t>(static_cast<std::uint32_t>(-1))) {
		throw std::runtime_error("Arena::put: value size exceeds maximum storable size");
	}
	if (value.size_bytes() >= segment_capacity_) {
		throw std::runtime_error("Arena::put: value size exceeds segment capacity");
	}

	// Reserve one bookkeeping byte ahead of the payload so stale handles can be rejected.
	const std::size_t required_bytes = value.size_bytes() + 1;

	if(current_segment_ == nullptr || current_segment_->capacity - current_segment_->used < required_bytes) {
		if(!freelist_.empty()) {
			current_segment_ = freelist_.back();
			freelist_.pop_back();
		} else {
			std::unique_ptr<Segment, SegmentDeleter> new_segment(
				static_cast<Segment*>(std::malloc(segment_capacity_ + sizeof(Segment))));
			assert(new_segment != nullptr && "Arena::put: segment allocation failed");
			new_segment->capacity = segment_capacity_;
			new_segment->used = 0;
			new_segment->live_allocations = 0;
			new_segment->generation = 1;
			heap_.push_back(std::move(new_segment));
			current_segment_ = heap_.back().get();
		}
	}

	Segment* segment = current_segment_;
	const std::uint32_t offset = segment->used;
	segment->bytes[offset] = std::byte{ 1 };
	std::memcpy(segment->bytes + offset + 1, value.data(), value.size_bytes());
	segment->used += required_bytes;
	segment->live_allocations++;

	return {
		.segment = segment,
		.offset = offset,
		.len = static_cast<std::uint32_t>(value.size_bytes()),
		.generation = segment->generation,
	};
}

std::span<const std::byte> Arena::view(const BufferHandle& handle) const noexcept {
	if(handle.segment == nullptr) {
		return { };
	}

	if(handle.segment->generation != handle.generation) {
		return { };
	}

	if(handle.segment->bytes[handle.offset] != std::byte{ 1 }) {
		return { };
	}

	return { handle.segment->bytes + handle.offset + 1, handle.len };
}

void Arena::take(BufferHandle&& handle) noexcept {
	if(handle.segment == nullptr) {
		return;
	}

	if(handle.segment->generation != handle.generation) {
		return;
	}

	if(handle.segment->bytes[handle.offset] != std::byte{ 1 }) {
		return;
	}

	handle.segment->bytes[handle.offset] = std::byte{ 0 };
	handle.segment->live_allocations--;
	if(handle.segment->live_allocations == 0) {
		handle.segment->used = 0;
		handle.segment->generation++;
		freelist_.push_back(handle.segment);
		if(current_segment_ == handle.segment) {
			current_segment_ = nullptr;
		}
	}
}

} // namespace scheduler
