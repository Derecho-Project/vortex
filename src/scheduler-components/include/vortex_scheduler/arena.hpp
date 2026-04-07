#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <vortex_scheduler/core.hpp>


#if __has_include(<boost/unordered/unordered_flat_map.hpp>)
#include <boost/unordered/unordered_flat_map.hpp>
template <typename K, typename V>
using ArenaRecordMap = boost::unordered_flat_map<K, V>;
#else
#include <boost/unordered_map.hpp>
template <typename K, typename V>
using ArenaRecordMap = boost::unordered_map<K, V>;
#endif

VORTEX_SCHEDULER_NAMESPACE_BEGIN

/**
 * @brief Opaque token identifying a live arena allocation.
 *
 * The handle can be stored and passed around safely, but it does not own the
 * underlying bytes. Callers must resolve the handle through ArenaAllocator to
 * access data. A handle becomes stale after the corresponding key is overwritten
 * or taken, or after its segment is recycled.
 */
struct BufferHandle {
  std::uint64_t allocation_id;
  std::size_t slot_id;
  std::uint64_t slot_version;
  std::size_t segment_id;
  std::uint64_t generation;
  std::size_t offset;
  std::size_t len;
};

/**
 * @brief Mutable non-owning view into bytes stored inside the arena.
 *
 * The view is only valid while the referenced BufferHandle remains live.
 */
struct BufferView {
  std::byte *bytes;
  std::size_t len;
};

/**
 * @brief Segmented bump allocator keyed by user-provided identifiers.
 *
 * Allocations are appended monotonically within the latest segment. Individual
 * frees do not reclaim space inside a segment; instead, a segment is recycled
 * only after all of its live allocations have been taken or overwritten.
 *
 * @tparam KeyType The lookup key associated with each stored byte buffer.
 */
template <typename KeyType> class ArenaAllocator {
public:
  /**
   * @brief Creates an arena with a preferred segment capacity.
   *
   * Segments may be larger than this value when a single allocation requires
   * more space.
   *
   * @param segment_capacity Preferred capacity, in bytes, for newly allocated
   *        segments.
   */
  explicit ArenaAllocator(std::size_t segment_capacity = kDefaultSegmentCapacity);

  /**
   * @brief Releases all arena-owned segment memory.
   */
  ~ArenaAllocator();

  ArenaAllocator(const ArenaAllocator &) = delete;
  ArenaAllocator &operator=(const ArenaAllocator &) = delete;

  /**
   * @brief Reserves metadata capacity for an expected steady-state workload.
   *
   * This reduces reallocation and rehashing overhead for long-running or
   * memory-rich deployments.
   *
   * @param expected_live_keys Expected number of concurrently live keys.
   * @param expected_live_allocations Expected number of concurrently live
   *        allocation slots.
   * @param expected_segments Expected number of segments likely to be active or
   *        recycled.
   */
  void reserve(std::size_t expected_live_keys,
               std::size_t expected_live_allocations = 0,
               std::size_t expected_segments = 0);

  /**
   * @brief Copies bytes into the arena and associates them with a key.
   *
   * If the key already exists, the previous allocation is implicitly freed. The
   * returned handle identifies the new live allocation.
   *
   * @param key Key to insert or replace.
   * @param data Bytes to copy into arena-managed storage.
   * @return Handle for the newly stored allocation.
   */
  BufferHandle put(const KeyType &key, std::span<const std::byte> data);

  /**
   * @brief Looks up the current handle for a key without removing it.
   *
   * @param key Key to query.
   * @return The current live handle for the key, or std::nullopt if absent.
   */
  std::optional<BufferHandle> get(const KeyType &key) const;

  /**
   * @brief Removes a key from the arena and invalidates its handle.
   *
   * Taking the last live allocation in a segment recycles that entire segment.
   *
   * @param key Key to remove.
   * @return The handle that used to be associated with the key, or std::nullopt
   *         if the key was not present.
   */
  std::optional<BufferHandle> take(const KeyType &key);

  /**
   * @brief Resolves a live handle to a mutable byte view.
   *
   * The returned bytes are owned by the arena and become invalid if the handle
   * becomes stale.
   *
   * @param handle Handle to resolve.
   * @return Mutable view into arena memory, or std::nullopt if the handle is
   *         stale.
   */
  std::optional<BufferView> resolve(const BufferHandle &handle);

  /**
   * @brief Resolves a live handle to a read-only byte view.
   *
   * @param handle Handle to resolve.
   * @return Const view into arena memory, or std::nullopt if the handle is
   *         stale.
   */
  std::optional<std::span<const std::byte>>
  resolve_const(const BufferHandle &handle) const;

private:
  /**
   * @brief Storage backing for one monotonic bump-allocation region.
   */
  struct Segment {
    std::byte *bytes = nullptr;
    std::size_t capacity = 0;
    std::size_t used = 0;
    std::size_t live_allocations = 0;
    std::uint64_t generation = 1;
  };

  /**
   * @brief Per-key metadata for the currently live allocation.
   */
  struct Record {
    BufferHandle handle;
  };

  /**
   * @brief Reusable slot storing validation metadata for a live allocation.
   *
   * Slot versions increase each time a slot is reused so stale handles can be
   * rejected even when slot indices are recycled.
   */
  struct LiveAllocation {
    const std::byte *data = nullptr;
    std::uint64_t allocation_id = 0;
    std::uint64_t slot_version = 0;
    std::size_t segment_id = 0;
    std::uint64_t generation = 0;
    std::size_t offset = 0;
    std::size_t len = 0;
    bool is_live = false;
  };

  static constexpr std::size_t kDefaultSegmentCapacity = 4096;

  std::size_t _segment_capacity;
  std::vector<Segment> _segments;
  std::vector<std::size_t> _recycled_segments;
  ArenaRecordMap<KeyType, Record> _records;
  std::vector<LiveAllocation> _live_allocations;
  std::vector<std::size_t> _free_live_slots;
  std::uint64_t _next_allocation_id = 1;

  BufferHandle allocate_and_copy(std::span<const std::byte> data);
  std::size_t acquire_segment(std::size_t min_capacity);
  void release_record(const Record &record);
  void recycle_segment(std::size_t segment_id);
  bool is_live_handle(const BufferHandle &handle) const;
};

VORTEX_SCHEDULER_NAMESPACE_END

#include "arena_impl.hpp"