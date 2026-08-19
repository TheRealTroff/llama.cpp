# Baseline: small-batch decode scaling on Metal (M4 Pro, macOS 26.5.2)

Commit: 6d054983 (build 1833), build dir `build-perf/` (Release, GGML_METAL=ON, embed library).

Command:
```
llama-bench -m <model> -fa 1 -ctk q8_0 -ctv q8_0 -n 0 -p 1,2,3,4,5,6,8 -r 3
```

ms per batch-N forward pass (= 1000*N/tps):

| N | 4B Q4_0 ms | x vs N=1 | 27B Q4_0 ms | x vs N=1 |
|---|-----------:|---------:|------------:|---------:|
| 1 |  13.7      | 1.00     |  80.3       | 1.00     |
| 2 |  18.4      | 1.34     | 126.7       | 1.58     |
| 3 |  25.8      | 1.88     | 173.2       | 2.16     |
| 4 |  33.3      | 2.43     | 226.8       | 2.82     |
| 5 |  45.6      | 3.33     | 335.1       | 4.17     |
| 6 |  47.1      | 3.44     | 330.0       | 4.11     |
| 8 |  63.7      | 4.65     | 437.4       | 5.45     |

Raw t/s (4B): 72.91, 108.75, 116.37, 120.21, 109.75, 127.50, 125.60
Raw t/s (27B): 12.45, 15.78, 17.32, 17.64, 14.92, 18.18, 18.29

Notes:
- Non-monotonic bump at N=5 present on both models (N=5 slower than N=6).
- Success criteria: 4B N=4 <= 1.6x (<= 21.9 ms), N=8 <= 2.5x (<= 34.2 ms), bit-identical logits.
