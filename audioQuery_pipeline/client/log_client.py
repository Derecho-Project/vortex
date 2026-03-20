#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
from derecho.cascade.external_client import ServiceClientAPI



if __name__ == "__main__":
    
    capi = ServiceClientAPI()
    
    
    # notify all nodes to flush logs
    # TODO: read it from object_pool.list to send this notification to both /rag/emb and /rag/generate object pool's subgroup's shards
    subgroup_type = "VolatileCascadeStoreWithStringKey"
    subgroup_index = 0
    num_shards = len(capi.get_subgroup_members(subgroup_type,subgroup_index))
    for i in range(num_shards):
        shard_index = i
        capi.put("/flush_log/hello", b"",subgroup_type=subgroup_type,
                 subgroup_index=subgroup_index,
                 shard_index=shard_index)
        print(f"Sent flush log notification to shard {shard_index} of subgroup {subgroup_index}")

    print("Done!")
