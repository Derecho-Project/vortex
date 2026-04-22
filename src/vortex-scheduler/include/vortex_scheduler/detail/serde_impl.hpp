#pragma once

#include <cassert>
#include <cstdint>

namespace scheduler::serde {

// ~~~ serde implementation ~~~

/// @brief align to the next offset that satisfies the alignment requirements of T
template <Serializable T>
static constexpr std::size_t pad_aligned_to(const std::size_t offset) noexcept {
	constexpr std::size_t align = alignof(T);
	return (align - (offset % align)) % align;
}

// ~~~ Sizer impl ~~~
constexpr std::size_t Sizer::bytes_written() const noexcept {
	return count;
}

template <Serializable T>
constexpr void Sizer::put() noexcept {
	count += sizeof(T);
}

template <Serializable T>
constexpr void Sizer::put_span(const std::span<T>& v) noexcept {
	count += pad_aligned_to<T>(count);
	count += v.size_bytes();
}

// ~~~ Writer impl ~~~

inline std::size_t Writer::remaining() const noexcept {
	return buf.size() - pos;
}

inline std::size_t Writer::bytes_written() const noexcept {
	return pos;
}

template <Serializable T>
void Writer::put(const T& v) noexcept {
	assert(remaining() >= sizeof(T) && "Writer::put: overflow");
	std::memcpy(buf.data() + pos, &v, sizeof(T));
	pos += sizeof(T);
}

template <Serializable T>
void Writer::put_span(const std::span<T>& v) noexcept {
	const std::size_t pad = pad_aligned_to<T>(pos);
	assert(remaining() >= pad + v.size_bytes() && "Writer::put_span: overflow");
	std::memset(buf.data() + pos, 0, pad); // zero pad to avoid leaking stack/heap
	pos += pad;

	assert(reinterpret_cast<std::uintptr_t>(buf.data() + pos) % alignof(T) == 0);
	std::memcpy(buf.data() + pos, v.data(), v.size_bytes());
	pos += v.size_bytes();
}

// ~~~ Reader impl ~~~
inline std::size_t Reader::remaining() const noexcept {
	return buf.size() - pos;
}

template <Serializable T>
T Reader::get() noexcept {
	assert(remaining() >= sizeof(T) && "Reader::get: buffer underrun");
	T v;
	std::memcpy(&v, buf.data() + pos, sizeof(T));
	pos += sizeof(T);
	return v;
}

template <Serializable T>
std::span<const T> Reader::get_span(std::size_t count) noexcept {
	const std::size_t pad = pad_aligned_to<T>(pos);
	const std::size_t bytes = sizeof(T) * count;
	assert(remaining() >= pad + bytes && "Reader::get_span: buffer underrun");
	const auto* base = buf.data() + pos + pad;
	assert(reinterpret_cast<std::uintptr_t>(base) % alignof(T) == 0 && "Reader::get_span: misaligned");
	pos += pad + bytes;
	return { reinterpret_cast<const T*>(base), count };
}

} // namespace scheduler::serde