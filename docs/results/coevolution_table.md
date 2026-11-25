| Condition | Run | Best speedup | Device-failure rate (early -> late) | Rules adopted | Rules rejected (corpus false positives) | Calibration version |
|---|---|---|---|---|---|---|
| fixed_harness | 0 | 1.250x | 4% -> 4% | - | - | v0 |
| fixed_harness | 1 | 1.266x | 6% -> 0% | - | - | v0 |
| co_evolving | 0 | 1.250x | 4% -> 0% | HW.TMEM_LANES | HW.TMEM_LANES_STRICT (kmeans_persistent_smem_8warps, kmeans_bn256_chunked) | v1 |
| co_evolving | 1 | 1.250x | 4% -> 0% | HW.TMEM_LANES | HW.TMEM_LANES_STRICT (kmeans_persistent_smem_8warps, kmeans_bn256_chunked) | v3 |
