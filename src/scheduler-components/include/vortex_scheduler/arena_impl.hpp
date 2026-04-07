#pragma once

#include <algorithm>
#include <cstring>
#include <new>
#include <utility>

#include <vortex_scheduler/core.hpp>
#ifdef __INTELLISENSE__
#include "arena.hpp"
#endif

VORTEX_SCHEDULER_NAMESPACE_BEGIN

template <typename KeyType>
ArenaAllocator<KeyType>::ArenaAllocator(std::size_t segment_capacity)
    : _segment_capacity(std::max(segment_capacity, std::size_t{1})) {}

template <typename KeyType>
void ArenaAllocator<KeyType>::reserve(std::size_t expected_live_keys,
                                      std::size_t expected_live_allocations,
                                      std::size_t expected_segments) {
  _records.reserve(expected_live_keys);
  if (expected_live_allocations > 0) {
    _live_allocations.reserve(expected_live_allocations);
    _free_live_slots.reserve(expected_live_allocations);
  }
  if (expected_segments > 0) {
    _segments.reserve(expected_segments);
    _recycled_segments.reserve(expected_segments);
  }
}

template <typename KeyType>
ArenaAllocator<KeyType>::~ArenaAllocator() {
  for (Segment &segment : _segments) {
    delete[] segment.bytes;
  }
}

template <typename KeyType>
BufferHandle ArenaAllocator<KeyType>::put(const KeyType &key,
                                          std::span<const std::byte> data) {
  auto [it, inserted] = _records.try_emplace(key);
  if (!inserted) {
    release_record(it->second);
  }

  BufferHandle handle = allocate_and_copy(data);
  it->second.handle = handle;
  return handle;
}

template <typename KeyType>
std::optional<BufferHandle>
ArenaAllocator<KeyType>::get(const KeyType &key) const {
  auto it = _records.find(key);
  if (it == _records.end()) {
    return std::nullopt;
  }

  return it->second.handle;
}

template <typename KeyType>
std::optional<BufferHandle> ArenaAllocator<KeyType>::take(const KeyType &key) {
  auto it = _records.find(key);
  if (it == _records.end()) {
    return std::nullopt;
  }

  BufferHandle handle = it->second.handle;
  release_record(it->second);
  _records.erase(it);
  return handle;
}

template <typename KeyType>
std::optional<BufferView>
ArenaAllocator<KeyType>::resolve(const BufferHandle &handle) {
  if (!is_live_handle(handle)) {
    return std::nullopt;
  }

  const LiveAllocation &live = _live_allocations[handle.slot_id];
  return BufferView{.bytes = const_cast<std::byte *>(live.data),
                    .len = live.len};
}

template <typename KeyType>
std::optional<std::span<const std::byte>>
ArenaAllocator<KeyType>::resolve_const(const BufferHandle &handle) const {
  if (!is_live_handle(handle)) {
    return std::nullopt;
  }

  const LiveAllocation &live = _live_allocations[handle.slot_id];
  return std::span<const std::byte>(live.data, live.len);
}

template <typename KeyType>
BufferHandle
ArenaAllocator<KeyType>::allocate_and_copy(std::span<const std::byte> data) {
  const std::size_t segment_id = acquire_segment(data.size());
  Segment &segment = _segments[segment_id];

  const std::size_t offset = segment.used;
  if (!data.empty()) {
    std::memcpy(segment.bytes + offset, data.data(), data.size());
  }

  const std::uint64_t allocation_id = _next_allocation_id++;
  std::size_t slot_id;
  if (_free_live_slots.empty()) {
    slot_id = _live_allocations.size();
    _live_allocations.push_back(LiveAllocation{});
  } else {
    slot_id = _free_live_slots.back();
    _free_live_slots.pop_back();
  }

  LiveAllocation &slot = _live_allocations[slot_id];
  ++slot.slot_version;
  slot.data = segment.bytes + offset;
  slot.allocation_id = allocation_id;
  slot.segment_id = segment_id;
  slot.generation = segment.generation;
  slot.offset = offset;
  slot.len = data.size();
  slot.is_live = true;

  segment.used += data.size();
  ++segment.live_allocations;

  return BufferHandle{.allocation_id = allocation_id,
                      .slot_id = slot_id,
                      .slot_version = slot.slot_version,
                      .segment_id = segment_id,
                      .generation = segment.generation,
                      .offset = offset,
                      .len = data.size()};
}

template <typename KeyType>
std::size_t ArenaAllocator<KeyType>::acquire_segment(std::size_t min_capacity) {
  if (!_segments.empty()) {
    const std::size_t latest_id = _segments.size() - 1;
    Segment &latest = _segments[latest_id];
    if (latest.capacity - latest.used >= min_capacity) {
      return latest_id;
    }
  }

  Segment segment;
  segment.capacity = std::max(_segment_capacity, min_capacity);
  segment.bytes = new (std::nothrow) std::byte[segment.capacity];

  if (segment.bytes != nullptr) {
    _segments.push_back(segment);
    return _segments.size() - 1;
  }

  for (auto it = _recycled_segments.begin(); it != _recycled_segments.end();
       ++it) {
    Segment &recycled = _segments[*it];
    if (recycled.capacity >= min_capacity) {
      const std::size_t segment_id = *it;
      _recycled_segments.erase(it);
      return segment_id;
    }
  }

  throw std::bad_alloc();
}

template <typename KeyType>
void ArenaAllocator<KeyType>::release_record(const Record &record) {
  if (record.handle.slot_id >= _live_allocations.size()) {
    return;
  }

  LiveAllocation &slot = _live_allocations[record.handle.slot_id];
  if (!slot.is_live || slot.slot_version != record.handle.slot_version ||
      slot.allocation_id != record.handle.allocation_id) {
    return;
  }

  if (slot.segment_id >= _segments.size()) {
    slot.is_live = false;
    _free_live_slots.push_back(record.handle.slot_id);
    return;
  }

  Segment &segment = _segments[slot.segment_id];
  if (segment.generation != slot.generation || segment.live_allocations == 0) {
    slot.is_live = false;
    _free_live_slots.push_back(record.handle.slot_id);
    return;
  }

  slot.is_live = false;
  _free_live_slots.push_back(record.handle.slot_id);
  --segment.live_allocations;
  if (segment.live_allocations == 0) {
    recycle_segment(slot.segment_id);
  }
}

template <typename KeyType>
void ArenaAllocator<KeyType>::recycle_segment(std::size_t segment_id) {
  Segment &segment = _segments[segment_id];
  segment.used = 0;
  ++segment.generation;
  _recycled_segments.push_back(segment_id);
}

template <typename KeyType>
bool ArenaAllocator<KeyType>::is_live_handle(const BufferHandle &handle) const {
  if (handle.allocation_id == 0 || handle.slot_id >= _live_allocations.size()) {
    return false;
  }

  const LiveAllocation &live = _live_allocations[handle.slot_id];
  if (!live.is_live) {
    return false;
  }

  if (live.slot_version != handle.slot_version ||
      live.allocation_id != handle.allocation_id ||
      live.segment_id != handle.segment_id ||
      live.generation != handle.generation || live.offset != handle.offset ||
      live.len != handle.len) {
    return false;
  }

  return true;
}

VORTEX_SCHEDULER_NAMESPACE_END