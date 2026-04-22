#include <catch2/catch.hpp>
#include <vortex_scheduler/dag_registry.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>

namespace scheduler {
namespace {

constexpr std::string_view UUID_A = "24e10f1c-1100-11eb-1111-0111ac110002";
constexpr std::string_view UUID_B = "a945d81b-53c7-43c3-b461-12c85d6883ab";
constexpr std::string_view UUID_C = "4f64e67a-44dd-4721-967d-d6039cc32470";
constexpr std::string_view UUID_D = "2bc87cc7-7ade-43d1-9a47-2a497b2fd5c0";
constexpr std::string_view UUID_OTHER = "ffffffff-ffff-ffff-ffff-ffffffffffff";

constexpr std::string_view DIAMOND_DFG = R"({
    "tasks": [
        { "task_id": 0, "pathname": "/A", "udl_uuid": "24e10f1c-1100-11eb-1111-0111ac110002" },
        { "task_id": 1, "pathname": "/B", "udl_uuid": "a945d81b-53c7-43c3-b461-12c85d6883ab" },
        { "task_id": 2, "pathname": "/C", "udl_uuid": "4f64e67a-44dd-4721-967d-d6039cc32470" },
        { "task_id": 3, "pathname": "/D", "udl_uuid": "2bc87cc7-7ade-43d1-9a47-2a497b2fd5c0" }
    ],
    "graphs": [
        {
            "graph_id": 0,
            "description": "canonical diamond DFG",
            "task_list": [0, 1, 2, 3],
            "upstream_by_task": [
                { "task_id": 0, "from": [] },
                { "task_id": 1, "from": [0] },
                { "task_id": 2, "from": [0] },
                { "task_id": 3, "from": [1, 2] }
            ]
        }
    ]
})";

// Write `body` to a fresh temporary .json file and return its path.
std::filesystem::path write_tmp_dfg(std::string_view body, std::string_view tag) {
    auto path = std::filesystem::temp_directory_path()
              / ("dag_registry_test_" + std::string(tag) + ".json");
    std::ofstream out(path, std::ios::trunc);
    out << body;
    return path;
}

std::set<uint16_t> as_set(std::span<const uint16_t> s) {
    return { s.begin(), s.end() };
}

} // namespace

TEST_CASE("DagRegistry loads diamond DFG", "[dag_registry]") {
    const auto path = write_tmp_dfg(DIAMOND_DFG, "diamond");
    DagRegistry reg(path, UUID_A);

    REQUIRE(reg.has_graph(0));
    REQUIRE_FALSE(reg.has_graph(1));

    SECTION("graph membership preserves load order") {
        const auto tasks = reg.graph_tasks(0);
        REQUIRE(tasks.size() == 4);
        CHECK(tasks[0] == 0);
        CHECK(tasks[1] == 1);
        CHECK(tasks[2] == 2);
        CHECK(tasks[3] == 3);
    }

    SECTION("upstream edges") {
        CHECK(reg.get_upstream_tasks(0, 0).empty());
        CHECK(as_set(reg.get_upstream_tasks(0, 1)) == std::set<uint16_t>{0});
        CHECK(as_set(reg.get_upstream_tasks(0, 2)) == std::set<uint16_t>{0});
        CHECK(as_set(reg.get_upstream_tasks(0, 3)) == std::set<uint16_t>{1, 2});
    }

    SECTION("downstream edges are derived correctly") {
        CHECK(as_set(reg.get_downstream_tasks(0, 0)) == std::set<uint16_t>{1, 2});
        CHECK(as_set(reg.get_downstream_tasks(0, 1)) == std::set<uint16_t>{3});
        CHECK(as_set(reg.get_downstream_tasks(0, 2)) == std::set<uint16_t>{3});
        CHECK(reg.get_downstream_tasks(0, 3).empty());
    }

    SECTION("unknown graph or task returns empty span") {
        CHECK(reg.get_upstream_tasks(99, 0).empty());
        CHECK(reg.get_downstream_tasks(99, 0).empty());
        CHECK(reg.get_upstream_tasks(0, 999).empty());
        CHECK(reg.get_downstream_tasks(0, 999).empty());
    }

    SECTION("find_node by id and pathname") {
        const auto* by_id = reg.find_node(uint16_t{2});
        REQUIRE(by_id != nullptr);
        CHECK(by_id->pathname == "/C");
        CHECK(by_id->udl_uuid == UUID_C);

        const auto* by_path = reg.find_node(std::string_view("/D"));
        REQUIRE(by_path != nullptr);
        CHECK(by_path->task_id == 3);
        CHECK(by_path->udl_uuid == UUID_D);

        CHECK(reg.find_node(uint16_t{42}) == nullptr);
        CHECK(reg.find_node(std::string_view("/missing")) == nullptr);
    }
}

TEST_CASE("DagRegistry filters graphs by udl_uuid", "[dag_registry]") {
    const auto path = write_tmp_dfg(DIAMOND_DFG, "filter");

    SECTION("matching UDL keeps the graph") {
        DagRegistry reg(path, UUID_B);
        CHECK(reg.has_graph(0));
    }

    SECTION("non-matching UDL drops the graph but keeps task catalog") {
        DagRegistry reg(path, UUID_OTHER);
        CHECK_FALSE(reg.has_graph(0));
        // Tasks are still indexed globally even when no graph matches.
        REQUIRE(reg.find_node(uint16_t{0}) != nullptr);
        CHECK(reg.find_node(uint16_t{0})->pathname == "/A");
    }

    SECTION("empty udl_uuid keeps all graphs") {
        DagRegistry reg(path, std::string_view{});
        CHECK(reg.has_graph(0));
    }
}

TEST_CASE("DagRegistry handles sparse / non-zero based task ids", "[dag_registry]") {
    constexpr std::string_view BODY = R"({
        "tasks": [
            { "task_id": 100, "pathname": "/x", "udl_uuid": "u-100" },
            { "task_id": 250, "pathname": "/y", "udl_uuid": "u-250" },
            { "task_id": 7,   "pathname": "/z", "udl_uuid": "u-7"   }
        ],
        "graphs": [
            {
                "graph_id": 5,
                "task_list": [100, 250, 7],
                "upstream_by_task": [
                    { "task_id": 250, "from": [100, 7] }
                ]
            }
        ]
    })";
    const auto path = write_tmp_dfg(BODY, "sparse");
    DagRegistry reg(path, std::string_view{});

    REQUIRE(reg.has_graph(5));
    CHECK(as_set(reg.get_upstream_tasks(5, 250)) == std::set<uint16_t>{100, 7});
    CHECK(as_set(reg.get_downstream_tasks(5, 100)) == std::set<uint16_t>{250});
    CHECK(as_set(reg.get_downstream_tasks(5, 7)) == std::set<uint16_t>{250});
    CHECK(reg.get_upstream_tasks(5, 100).empty());
    CHECK(reg.get_upstream_tasks(5, 7).empty());

    // task_id between min and max but not in the graph -> empty span.
    CHECK(reg.get_upstream_tasks(5, 200).empty());
    CHECK(reg.get_downstream_tasks(5, 200).empty());
}

TEST_CASE("DagRegistry supports multiple graphs sharing tasks", "[dag_registry]") {
    constexpr std::string_view BODY = R"({
        "tasks": [
            { "task_id": 0, "pathname": "/A", "udl_uuid": "ua" },
            { "task_id": 1, "pathname": "/B", "udl_uuid": "ub" },
            { "task_id": 2, "pathname": "/C", "udl_uuid": "uc" }
        ],
        "graphs": [
            {
                "graph_id": 1,
                "task_list": [0, 1, 2],
                "upstream_by_task": [
                    { "task_id": 1, "from": [0] },
                    { "task_id": 2, "from": [1] }
                ]
            },
            {
                "graph_id": 2,
                "task_list": [0, 2],
                "upstream_by_task": [
                    { "task_id": 2, "from": [0] }
                ]
            }
        ]
    })";
    const auto path = write_tmp_dfg(BODY, "multi");
    DagRegistry reg(path, std::string_view{});

    REQUIRE(reg.has_graph(1));
    REQUIRE(reg.has_graph(2));

    // Same task id, different topology per graph.
    CHECK(as_set(reg.get_upstream_tasks(1, 2)) == std::set<uint16_t>{1});
    CHECK(as_set(reg.get_upstream_tasks(2, 2)) == std::set<uint16_t>{0});
    CHECK(as_set(reg.get_downstream_tasks(1, 0)) == std::set<uint16_t>{1});
    CHECK(as_set(reg.get_downstream_tasks(2, 0)) == std::set<uint16_t>{2});
    // Task 1 is not in graph 2.
    CHECK(reg.get_upstream_tasks(2, 1).empty());
    CHECK(reg.get_downstream_tasks(2, 1).empty());
}

TEST_CASE("DagRegistry rejects malformed inputs", "[dag_registry]") {
    SECTION("missing file") {
        REQUIRE_THROWS_AS(
            DagRegistry(std::filesystem::path("/no/such/file_xyz.json"), UUID_A),
            std::runtime_error);
    }

    SECTION("invalid JSON") {
        const auto path = write_tmp_dfg("{ not valid json", "bad_json");
        REQUIRE_THROWS_AS(DagRegistry(path, UUID_A), std::runtime_error);
    }

    SECTION("missing top-level fields") {
        const auto path = write_tmp_dfg(R"({"tasks":[]})", "no_graphs");
        REQUIRE_THROWS_AS(DagRegistry(path, UUID_A), std::runtime_error);
    }

    SECTION("duplicate task_id") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [
                { "task_id": 1, "pathname": "/a", "udl_uuid": "u" },
                { "task_id": 1, "pathname": "/b", "udl_uuid": "u" }
            ],
            "graphs": []
        })", "dup_id");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }

    SECTION("duplicate pathname") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [
                { "task_id": 1, "pathname": "/same", "udl_uuid": "u" },
                { "task_id": 2, "pathname": "/same", "udl_uuid": "u" }
            ],
            "graphs": []
        })", "dup_path");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }

    SECTION("graph references unknown task_id") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [{ "task_id": 1, "pathname": "/a", "udl_uuid": "u" }],
            "graphs": [{ "graph_id": 0, "task_list": [1, 99] }]
        })", "unknown_tid");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }

    SECTION("duplicate task_id in task_list") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [{ "task_id": 1, "pathname": "/a", "udl_uuid": "u" }],
            "graphs": [{ "graph_id": 0, "task_list": [1, 1] }]
        })", "dup_in_list");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }

    SECTION("upstream references task not in task_list") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [
                { "task_id": 1, "pathname": "/a", "udl_uuid": "u" },
                { "task_id": 2, "pathname": "/b", "udl_uuid": "u" }
            ],
            "graphs": [{
                "graph_id": 0,
                "task_list": [1],
                "upstream_by_task": [{ "task_id": 1, "from": [2] }]
            }]
        })", "up_outside");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }

    SECTION("self-loop") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [{ "task_id": 1, "pathname": "/a", "udl_uuid": "u" }],
            "graphs": [{
                "graph_id": 0,
                "task_list": [1],
                "upstream_by_task": [{ "task_id": 1, "from": [1] }]
            }]
        })", "self_loop");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }

    SECTION("duplicate graph_id") {
        const auto path = write_tmp_dfg(R"({
            "tasks": [{ "task_id": 1, "pathname": "/a", "udl_uuid": "u" }],
            "graphs": [
                { "graph_id": 0, "task_list": [1] },
                { "graph_id": 0, "task_list": [1] }
            ]
        })", "dup_graph");
        REQUIRE_THROWS_AS(DagRegistry(path, std::string_view{}), std::runtime_error);
    }
}

} // namespace scheduler