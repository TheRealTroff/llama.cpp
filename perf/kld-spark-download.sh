#!/bin/bash
# resumable parallel shard download of Qwen/Qwen3.8-27B; HF_TOKEN optional (export before running)
cd ~/kld/models/Qwen3.8-27B || exit 1
B=https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
export B; export AUTHHDR="${AUTH[*]}"
{ for i in $(seq -w 1 18); do echo model-000${i}-of-00018.safetensors; done
  for f in model.safetensors.index.json config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json chat_template.jinja; do echo $f; done; } \
| xargs -P 6 -I{} bash -c 'curl -sS -L -C - --retry 20 --retry-delay 5 $AUTHHDR -o "{}" "$B/{}" && echo "done {} $(date +%H:%M:%S)"'
echo ALL_DONE $(date +%H:%M:%S)
