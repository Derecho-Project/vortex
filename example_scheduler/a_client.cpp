#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cascade/cascade.hpp>
#include <cascade/service_client_api.hpp>

#include "scheduler_messages.hpp"
#include <vortex_scheduler/messages.hpp>

using namespace derecho::cascade;

namespace {

void print_usage(const char* argv0) {
	std::cout << "Usage: " << argv0 << " <job_start> <count> [sleep_ms=0] [graph_id=0]"
			  << " [target_task_id=0] [stream_id=0]"
			  << " [subgroup_index=0] [shard_index=0]" << std::endl;
}

// Builds a TaskOutput wire packet destined for `target_task` with an empty
// dye list and a serialized StepAMessage payload. Task A is the entry point of
// the diamond DFG, so the source_task is set equal to the target_task
// (self-edge) - the join thread will treat the client as a synthetic upstream.
std::vector<std::byte> build_task_output_packet(std::uint16_t graph_id,
												std::uint16_t job_id,
												std::uint16_t target_task_id,
												std::uint16_t stream_id,
												const StepAMessage& step_a_message) {
	const scheduler::message::TaskRef target_ref{
		.graph_id = graph_id,
		.job_id = job_id,
		.task_id = target_task_id,
		.stream_id = stream_id,
	};

	const scheduler::message::Decision decision{
		.source_worker = scheduler::message::WorkerRef{ .shard_number = 0 },
		.target_worker = scheduler::message::WorkerRef{ .shard_number = 0 },
		.source_task = target_ref, // self-edge: client acts as the upstream of A
		.target_task = target_ref,
	};

	std::vector<std::byte> step_a_payload(step_a_message.size_estimate());
	const std::size_t step_a_written = step_a_message.to_buffer(step_a_payload);
	step_a_payload.resize(step_a_written);

	scheduler::message::TaskOutput task_output{
		.decision = decision,
		.payload = std::move(step_a_payload),
	};

	std::vector<std::byte> buffer(task_output.size_estimate());
	const std::size_t written = task_output.to_buffer(buffer);
	buffer.resize(written);
	return buffer;
}

} // namespace

int main(int argc, char** argv) {
	if(argc < 3) {
		print_usage(argv[0]);
		return -1;
	}

	const std::uint32_t job_start = static_cast<std::uint32_t>(std::stoul(argv[1]));
	const std::uint32_t count = static_cast<std::uint32_t>(std::stoul(argv[2]));
	const std::uint32_t sleep_ms = (argc >= 4) ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 0;
	const std::uint16_t graph_id = (argc >= 5) ? static_cast<std::uint16_t>(std::stoul(argv[4])) : 0;
	const std::uint16_t target_task_id = (argc >= 6) ? static_cast<std::uint16_t>(std::stoul(argv[5])) : 0;
	const std::uint16_t stream_id = (argc >= 7) ? static_cast<std::uint16_t>(std::stoul(argv[6])) : 0;
	const std::uint32_t subgroup_index = (argc >= 8) ? static_cast<std::uint32_t>(std::stoul(argv[7])) : 0;
	const std::uint32_t shard_index = (argc >= 9) ? static_cast<std::uint32_t>(std::stoul(argv[8])) : 0;

	auto& capi = ServiceClientAPI::get_service_client();

	for(std::uint32_t i = 0; i < count; ++i) {
		const std::string payload_text = "Ingress Message to A #" + std::to_string(job_start + i);

		StepAMessage message{
			.metadata = MetadataA{
				.example_field4 = static_cast<double>(job_start + i),
				.example_field3 = static_cast<float>(i),
				.example_field1 = job_start + i,
				.example_field2 = 'A',
				.example_field5 = true,
			},
			.message = payload_text,
		};

		auto packet = build_task_output_packet(
			graph_id,
			static_cast<std::uint16_t>(job_start + i),
			target_task_id,
			stream_id,
			message);

		ObjectWithStringKey obj;
		obj.key = "/A/" + std::to_string(job_start + i);
		const std::size_t total_size = packet.size();
		obj.blob = Blob(
			[data = std::move(packet)](std::uint8_t* out, std::size_t) mutable -> std::size_t {
				if(!data.empty()) {
					std::memcpy(out, data.data(), data.size());
				}
				return data.size();
			},
			total_size);

		capi.put_and_forget<VolatileCascadeStoreWithStringKey>(obj, subgroup_index, shard_index, true);

		std::cout << "sent job_id=" << (job_start + i) << " key=" << obj.key << " bytes=" << total_size
				  << " subgroup=" << subgroup_index << " shard=" << shard_index << std::endl;

		if(sleep_ms > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
		}
	}

	return 0;
}
