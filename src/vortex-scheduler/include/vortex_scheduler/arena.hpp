/** Allocator for holding arbitrary trivially constructible types */

#pragma once

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace scheduler {

/// @brief thread safe arena allocator for arbitrary buffers
class Arena {
public:
	/// @brief Creates an arena that allocates segments with a preferred capacity.
	explicit Arena(std::size_t segment_capacity);
	Arena() = delete;

	Arena(const Arena& arena) = delete;
	Arena& operator=(const Arena& arena) = delete;

	Arena(Arena&& arena) = delete;
	Arena& operator=(Arena&& arena) = delete;

	~Arena() = default;

public:
	struct Segment {
		std::uint32_t capacity;
		std::uint32_t used;
		std::uint32_t live_allocations;
		std::uint8_t generation;
		std::byte bytes[]; // flexible array member for segment storage
	};

	struct SegmentDeleter {
		void operator()(Segment* segment) const noexcept {
			std::free(segment);
		}
	};

	struct BufferHandle {
		Segment* segment;
		std::uint32_t offset;
		std::uint32_t len;
		std::uint8_t generation;
	};

	/**
     * @brief Allocates a value in the arena and returns a handle to it. O(n) on value size.
     *
     * @param value Value to store in the arena. 
     * @return Handle that can be resolved to access the stored value.
     */
	BufferHandle put(const std::span<const std::byte>& value);

	/**
     * @brief Resolves a live handle to a immutable byte view. UB if handle is stale. O(1).
     *
     * @param handle Handle to resolve.
     * @return Immutable view into arena memory
     */
	std::span<const std::byte> view(const BufferHandle& handle) const noexcept;

	/**
     * @brief Removes a handle from the arena and invalidates it. UB if handle is stale. O(1).
     */
	void take(BufferHandle&& handle) noexcept;

private:
	std::size_t segment_capacity_;
	std::vector<std::unique_ptr<Segment, SegmentDeleter>> heap_;
	std::vector<Segment*> freelist_;
	Segment* current_segment_ = nullptr;
};

} // namespace scheduler