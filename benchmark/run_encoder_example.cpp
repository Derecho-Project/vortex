/** Just a simple example of how to start at the encoder udl instead of centroid 
 * search udl :)
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <cascade/service_client_api.hpp>

#include "../vortex_udls/serialize_utils.hpp"

using namespace derecho::cascade;

#define UDL1_PATH "/rag/emb/encode_search"
#define UDL2_PATH "/rag/emb/centroids_search"
#define UDL3_PATH "/rag/emb/clusters_search"
#define UDL4_PATH "/rag/generate/agg"

#define UDL1_TIMESTAMP_FILE "udl1.dat"
#define UDL2_TIMESTAMP_FILE "udl2.dat"
#define UDL3_TIMESTAMP_FILE "udl3.dat"
#define UDL4_TIMESTAMP_FILE "udl4.dat" 

#define UDL1_SUBGROUP_INDEX 0
#define UDL2_SUBGROUP_INDEX 1
#define UDL3_SUBGROUP_INDEX 2
#define UDL4_SUBGROUP_INDEX 3
#define UDLS_SUBGROUP_TYPE VolatileCascadeStoreWithStringKey 

#define VORTEX_CLIENT_MAX_WAIT_TIME 60

#define UDL3_SUBGROUP_INDEX 2

const int ID = 0;
ServiceClientAPI& capi = ServiceClientAPI::get_service_client();

int main() {

    std::vector<std::pair<query_id_t, std::shared_ptr<std::string>>> queries = {
        {0 ,std::make_shared<std::string>("hello this is query 1")},
        {1 ,std::make_shared<std::string>("and this is query 2")},
        {2 ,std::make_shared<std::string>("What's the weather today?")},
        {4 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {5 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {6 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {7 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {8 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {9 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {10 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {11 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {12 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {13 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {14 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {15 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {16 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {17 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {18 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {19 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {20 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {21 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {22 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {23 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {24 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {25 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {26 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {27 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {28 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {29 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
        {30 ,std::make_shared<std::string>("SUPER LONG MESSAGE kjsdjflkasfj;asdkljf;klasdfj;aosdklfj;asdlkfjl;asdkfjl;asdfjk;asldkfj;asldkfjasl;dfjkl;asdkfjl;asdkfjl;askdfjl;aksdjfl;adksjfl;asdkjf;alskdfj;lakdjf;")},
    };

    EncoderQueryBatcher batcher(1024, queries.size());

    for(const auto& query : queries) {
        batcher.add_query(
            query.first,
            ID,
            query.second
        );
    }

    batcher.serialize();
    ObjectWithStringKey obj;
    obj.key = UDL1_PATH "/" + std::string("batch1");
    obj.blob = std::move(*batcher.get_blob());
    capi.trigger_put(obj);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return 0;
}