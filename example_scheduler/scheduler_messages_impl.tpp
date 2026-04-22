// #ifdef __INTELLISENSE__
// #include "scheduler_messages.hpp"
// #endif

#include <vortex_scheduler/serde.hpp>

namespace derecho::cascade {

// Wire format:
//   [Metadata POD][uint32_t length][char[length] message bytes]
// The length prefix is fixed-width (uint32_t) so messages are portable across
// 32/64-bit peers.

template <typename Family, typename Tag>
StepAMessageT<Family, Tag> StepAMessageT<Family, Tag>::from_buffer(std::span<const std::byte> payload) {
	scheduler::serde::Reader reader{ payload };

	const MetadataA metadata = reader.get<MetadataA>();
	const auto length = reader.get<std::uint32_t>();
	const auto string_view = reader.get_span<char>(length);
	return StepAMessageT<Family, Tag>{
		.metadata = metadata,
		.message  = string_repr_t::make(string_view),
	};
}

template <typename Family, typename Tag>
std::size_t StepAMessageT<Family, Tag>::to_buffer(const std::span<std::byte>& buffer) const noexcept {
	scheduler::serde::Writer writer{ buffer };
	const auto message_span = string_repr_t::view(message);

	writer.put<MetadataA>(metadata);
	writer.put<std::uint32_t>(static_cast<std::uint32_t>(message_span.size()));
	writer.put_span(message_span);
	return writer.bytes_written();
}

template <typename Family, typename Tag>
std::size_t StepAMessageT<Family, Tag>::size_estimate() const noexcept {
	scheduler::serde::Sizer sizer{ };
	sizer.put<MetadataA>();
	sizer.put<std::uint32_t>();
	sizer.put_span<const char>(string_repr_t::view(message));
	return sizer.bytes_written();
}

} // namespace derecho::cascade