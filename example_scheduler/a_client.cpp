#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cascade/cascade.hpp>
#include <cascade/service_client_api.hpp>

#include <vortex_scheduler/prelude.hpp>

#include "diamond_messages.hpp"

using namespace derecho::cascade;

namespace {

void print_usage(const char* argv0) {
	std::cout << "Usage: " << argv0
			  << " <job_start> <count> [sleep_ms=0] [graph_id=0] [source_task_id=0]"
			  << " [subgroup_index=0] [shard_index=0]"
			  << std::endl;
}

std::vector<uint8_t> make_task_output_packet(uint16_t job_id,
										 uint16_t graph_id,
										 uint16_t source_task_id,
										 const std::string& message) {
	StepAMessage msg {message};
	std::vector<uint8_t> payload(msg.size_estimate());
	msg.to_bytes(payload.data());

	scheduler::TaskOutput header;
	header.worker_id = 0;
	header.job_id = job_id;
	header.target_task_id = 0; // Task A
	header.source_task_id = source_task_id;
	header.graph_id = graph_id;
	header.payload_size = static_cast<uint32_t>(payload.size());

	const size_t header_size = header.size_estimate();
	std::vector<uint8_t> packet(header_size + payload.size());
	header.to_bytes(packet.data());
	if(!payload.empty()) {
		std::memcpy(packet.data() + header_size, payload.data(), payload.size());
	}

	return packet;
}

} // namespace

int main(int argc, char** argv) {
	if(argc < 3) {
		print_usage(argv[0]);
		return -1;
	}

	const uint32_t job_start = static_cast<uint32_t>(std::stoul(argv[1]));
	const uint32_t count = static_cast<uint32_t>(std::stoul(argv[2]));
	const uint32_t sleep_ms = (argc >= 4) ? static_cast<uint32_t>(std::stoul(argv[3])) : 0;
	const uint16_t graph_id = (argc >= 5) ? static_cast<uint16_t>(std::stoul(argv[4])) : 0;
	const uint16_t source_task_id = (argc >= 6) ? static_cast<uint16_t>(std::stoul(argv[5])) : 0;
	const uint32_t subgroup_index = (argc >= 7) ? static_cast<uint32_t>(std::stoul(argv[6])) : 0;
	const uint32_t shard_index = (argc >= 8) ? static_cast<uint32_t>(std::stoul(argv[7])) : 0;

	auto& capi = ServiceClientAPI::get_service_client();

	for(uint32_t i = 0; i < count; ++i) {
		const uint16_t job_id = static_cast<uint16_t>(job_start + i);
		const std::string message = "client->A job " + std::to_string(job_id);
		auto packet = make_task_output_packet(job_id, graph_id, source_task_id, message);

		ObjectWithStringKey obj;
		obj.key = "/A/" + std::to_string(job_id);
		const size_t total_size = packet.size();
		obj.blob = Blob(
			[data = std::move(packet)](uint8_t* out, std::size_t) mutable -> std::size_t {
				if(!data.empty()) {
					std::memcpy(out, data.data(), data.size());
				}
				return data.size();
			},
			total_size);

		capi.put_and_forget<VolatileCascadeStoreWithStringKey>(
			obj, subgroup_index, shard_index, true);
		std::cout << "sent job_id=" << job_id << " key=" << obj.key
				  << " subgroup=" << subgroup_index
				  << " shard=" << shard_index << std::endl;

		if(sleep_ms > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
		}
	}

	return 0;
}
