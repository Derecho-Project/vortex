#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <cascade/cascade.hpp>
#include <cascade/service_client_api.hpp>

#include <vortex_scheduler/prelude.hpp>

using namespace derecho::cascade;

namespace {

void print_usage(const char* argv0) {
	std::cout << "Usage: " << argv0
			  << " <job_start> <count> [worker_id=0] [graph_id=0]"
			  << " [subgroup_index=0] [shard_index=0]" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
	if(argc < 3) {
		print_usage(argv[0]);
		return -1;
	}

	const uint32_t job_start = static_cast<uint32_t>(std::stoul(argv[1]));
	const uint32_t count = static_cast<uint32_t>(std::stoul(argv[2]));
	const uint16_t worker_id = (argc >= 4) ? static_cast<uint16_t>(std::stoul(argv[3])) : 0;
	const uint16_t graph_id = (argc >= 5) ? static_cast<uint16_t>(std::stoul(argv[4])) : 0;
	const uint32_t subgroup_index = (argc >= 6) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;
	const uint32_t shard_index = (argc >= 7) ? static_cast<uint32_t>(std::stoul(argv[6])) : 0;

	auto& capi = ServiceClientAPI::get_service_client();

	for(uint32_t i = 0; i < count; ++i) {
		const uint16_t job_id = static_cast<uint16_t>(job_start + i);

		scheduler::TaskOutput trigger;
		trigger.worker_id = worker_id;
		trigger.job_id = job_id;
		trigger.target_task_id = 0; // /A
		trigger.source_task_id = 0;
		trigger.graph_id = graph_id;
		trigger.payload_size = 0;

		const size_t header_size = trigger.size_estimate();
		ObjectWithStringKey obj;
		obj.key = "/A/" + std::to_string(job_id);
		obj.blob = Blob(
			[trigger](uint8_t* out, std::size_t) mutable -> std::size_t {
				trigger.to_bytes(out);
				return trigger.size_estimate();
			},
			header_size);

		capi.put_and_forget<VolatileCascadeStoreWithStringKey>(obj, subgroup_index, shard_index, true);

		std::cout << "bootstrapped job=" << job_id
				  << " worker=" << worker_id
				  << " graph=" << graph_id << std::endl;
	}

	return 0;
}
