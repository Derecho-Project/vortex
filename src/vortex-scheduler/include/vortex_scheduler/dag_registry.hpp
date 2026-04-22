#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace scheduler {

/// @brief dag topologies loaded from a DFG file
class DagRegistry {
public:
    struct Node {
        uint16_t    task_id;
        std::string pathname;
        std::string udl_uuid;
    };

    DagRegistry() = delete;
    DagRegistry(const std::filesystem::path& dfg_path, std::string_view udl_uuid);

    /// @brief finds the upstream dependencies of a task. Returns empty span if `graph_id` or `task_id` is unknown.
    std::span<const uint16_t> get_upstream_tasks(uint16_t graph_id,
                                                 uint16_t task_id) const noexcept;

    /// @brief finds the downstream dependents of a task. Returns empty span if `graph_id` or `task_id` is unknown.
    std::span<const uint16_t> get_downstream_tasks(uint16_t graph_id,
                                                   uint16_t task_id) const noexcept;

    /// @brief finds a node by its task_id
    const Node* find_node(uint16_t task_id)          const noexcept;

    /// @brief finds a node by its pathname
    const Node* find_node(std::string_view pathname) const noexcept;

    /// @brief returns true if `graph_id` is known in the registry
    bool has_graph(uint16_t graph_id) const noexcept;
    
    /// @brief tasks in `graph_id` in the order they were listed in the DFG file. Returns empty span if `graph_id` is unknown.
    std::span<const uint16_t> graph_tasks(uint16_t graph_id) const noexcept;

private:
    // CSR adjacency keyed by task_id: edges live in
    //   [offsets[task_id], offsets[task_id + 1])
    // `offsets` is sized to (max_task_id_in_graph + 2). Task ids not present
    // in the graph have offsets[i] == offsets[i+1] -> empty span.
    struct Graph {
        std::vector<uint16_t> task_ids;          // membership, load order
        std::vector<uint32_t> upstream_offsets;
        std::vector<uint16_t> upstream_edges;
        std::vector<uint32_t> downstream_offsets;
        std::vector<uint16_t> downstream_edges;
    };

    // Tasks owned once; graphs reference them by id.
    std::vector<Node>                              tasks_;
    std::unordered_map<uint16_t, uint32_t>         task_index_by_id_;
    std::unordered_map<std::string_view, uint32_t> task_index_by_pathname_;

    std::unordered_map<uint16_t, Graph>            graphs_;
};

} // namespace scheduler