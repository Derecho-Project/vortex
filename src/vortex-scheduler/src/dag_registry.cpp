#include <vortex_scheduler/dag_registry.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace scheduler {

namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("DagRegistry: " + msg);
}

} // namespace

DagRegistry::DagRegistry(const std::filesystem::path& dfg_path,
                         std::string_view udl_uuid) try {
    std::ifstream in(dfg_path);
    if(!in.is_open()) {
        fail("could not open DFG file: " + dfg_path.string());
    }

    const json root = json::parse(in);

    // ---- 1. Load tasks --------------------------------------------------
    // Reserve up front so string_views into Node::pathname remain stable.
    const auto& tasks_json = root.at("tasks");
    tasks_.reserve(tasks_json.size());

    for(const auto& tj : tasks_json) {
        Node node{
            .task_id  = tj.at("task_id").get<uint16_t>(),
            .pathname = tj.at("pathname").get<std::string>(),
            .udl_uuid = tj.at("udl_uuid").get<std::string>(),
        };

        if(task_index_by_id_.contains(node.task_id)) {
            fail("duplicate task_id: " + std::to_string(node.task_id));
        }
        if(task_index_by_pathname_.contains(node.pathname)) {
            fail("duplicate pathname: " + node.pathname);
        }

        const auto idx = static_cast<uint32_t>(tasks_.size());
        tasks_.push_back(std::move(node));
        const auto& stored = tasks_.back();
        task_index_by_id_.emplace(stored.task_id, idx);
        task_index_by_pathname_.emplace(std::string_view(stored.pathname), idx);
    }

    // ---- 2. Load graphs -------------------------------------------------
    for(const auto& gj : root.at("graphs")) {
        const auto graph_id = gj.at("graph_id").get<uint16_t>();
        const auto task_ids = gj.at("task_list").get<std::vector<uint16_t>>();

        std::unordered_set<uint16_t> task_id_set(task_ids.begin(), task_ids.end());
        if(task_id_set.size() != task_ids.size()) {
            fail("graph " + std::to_string(graph_id) + " has duplicate task_ids in 'task_list'");
        }

        bool matches_udl = udl_uuid.empty();
        uint16_t max_task_id = 0;
        for(const auto tid : task_ids) {
            const auto it = task_index_by_id_.find(tid);
            if(it == task_index_by_id_.end()) {
                fail("graph " + std::to_string(graph_id) + " references unknown task_id "
                     + std::to_string(tid));
            }
            max_task_id = std::max(max_task_id, tid);
            matches_udl = matches_udl || tasks_[it->second].udl_uuid == udl_uuid;
        }

        if(!matches_udl) continue;  // skip graphs unrelated to this UDL

        // upstream_by_task[tid] = list of upstream task_ids for tid
        std::unordered_map<uint16_t, std::vector<uint16_t>> upstream_by_task;
        upstream_by_task.reserve(task_ids.size());

        for(const auto& entry : gj.value("upstream_by_task", json::array())) {
            const auto tid = entry.at("task_id").get<uint16_t>();
            const auto from = entry.at("from").get<std::vector<uint16_t>>();

            if(!task_id_set.contains(tid)) {
                fail("graph " + std::to_string(graph_id)
                     + " upstream_by_task references task not in task_list: " + std::to_string(tid));
            }

            std::unordered_set<uint16_t> seen;
            auto& list = upstream_by_task[tid];
            list.reserve(from.size());
            for(const auto up : from) {
                if(up == tid) {
                    fail("graph " + std::to_string(graph_id) + " task " + std::to_string(tid)
                         + " lists itself as upstream");
                }
                if(!task_id_set.contains(up)) {
                    fail("graph " + std::to_string(graph_id) + " task " + std::to_string(tid)
                         + " upstream " + std::to_string(up) + " is not in task_list");
                }
                if(seen.insert(up).second) list.push_back(up);
            }
        }

        // ---- 2a. Build CSR adjacency keyed by task_id -------------------
        // offsets sized to (max_task_id + 2) so [tid+1] is always valid.
        Graph g;
        g.task_ids = task_ids;

        const uint32_t offsets_size = static_cast<uint32_t>(max_task_id) + 2u;
        g.upstream_offsets.assign(offsets_size, 0u);
        g.downstream_offsets.assign(offsets_size, 0u);

        // Counts: stash in offsets[tid+1] then prefix-sum.
        uint32_t total_edges = 0;
        for(const auto& [tid, ups] : upstream_by_task) {
            g.upstream_offsets[tid + 1] = static_cast<uint32_t>(ups.size());
            total_edges += static_cast<uint32_t>(ups.size());
            for(const auto up : ups) {
                g.downstream_offsets[up + 1] += 1;
            }
        }
        for(uint32_t i = 1; i < offsets_size; ++i) {
            g.upstream_offsets[i]   += g.upstream_offsets[i - 1];
            g.downstream_offsets[i] += g.downstream_offsets[i - 1];
        }

        g.upstream_edges.resize(total_edges);
        g.downstream_edges.resize(total_edges);

        // Scatter using per-row cursors initialized from offsets.
        std::vector<uint32_t> up_cursor   = g.upstream_offsets;
        std::vector<uint32_t> down_cursor = g.downstream_offsets;
        for(const auto& [tid, ups] : upstream_by_task) {
            for(const auto up : ups) {
                g.upstream_edges[up_cursor[tid]++]    = up;
                g.downstream_edges[down_cursor[up]++] = tid;
            }
        }

        if(!graphs_.emplace(graph_id, std::move(g)).second) {
            fail("duplicate graph_id: " + std::to_string(graph_id));
        }
    }
} catch(const json::exception& e) {
    fail(std::string("JSON error in ") + dfg_path.string() + ": " + e.what());
}

std::span<const uint16_t>
DagRegistry::get_upstream_tasks(uint16_t graph_id, uint16_t task_id) const noexcept {
    const auto it = graphs_.find(graph_id);
    if(it == graphs_.end()) return {};
    const auto& g = it->second;
    if(static_cast<size_t>(task_id) + 1 >= g.upstream_offsets.size()) return {};
    const auto begin = g.upstream_offsets[task_id];
    const auto end   = g.upstream_offsets[task_id + 1];
    return std::span<const uint16_t>(g.upstream_edges.data() + begin, end - begin);
}

std::span<const uint16_t>
DagRegistry::get_downstream_tasks(uint16_t graph_id, uint16_t task_id) const noexcept {
    const auto it = graphs_.find(graph_id);
    if(it == graphs_.end()) return {};
    const auto& g = it->second;
    if(static_cast<size_t>(task_id) + 1 >= g.downstream_offsets.size()) return {};
    const auto begin = g.downstream_offsets[task_id];
    const auto end   = g.downstream_offsets[task_id + 1];
    return std::span<const uint16_t>(g.downstream_edges.data() + begin, end - begin);
}

const DagRegistry::Node* DagRegistry::find_node(uint16_t task_id) const noexcept {
    const auto it = task_index_by_id_.find(task_id);
    return it == task_index_by_id_.end() ? nullptr : &tasks_[it->second];
}

const DagRegistry::Node* DagRegistry::find_node(std::string_view pathname) const noexcept {
    const auto it = task_index_by_pathname_.find(pathname);
    return it == task_index_by_pathname_.end() ? nullptr : &tasks_[it->second];
}

bool DagRegistry::has_graph(uint16_t graph_id) const noexcept {
    return graphs_.contains(graph_id);
}

std::span<const uint16_t> DagRegistry::graph_tasks(uint16_t graph_id) const noexcept {
    const auto it = graphs_.find(graph_id);
    if(it == graphs_.end()) return {};
    return std::span<const uint16_t>(it->second.task_ids);
}

} // namespace scheduler
