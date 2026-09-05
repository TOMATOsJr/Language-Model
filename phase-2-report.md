training/checkpoints/best2.pt   --output hindi/training/checkpoints/intrinsic_metrics.json --max-tokens 5000000
Evaluating validation: 100%|███████████████████████████████████████████████████████████████████| 306/306 [34:47<00:00,  6.82s/batch]
validation: loss=2.971669 ppl=19.5245 bpb=0.485624
Evaluating test: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [35:22<00:00,  6.93s/batch]
test: loss=2.981799 ppl=19.7233 bpb=0.487999

python -m nepali.training.intrinsic_metrics   --checkpoint nepali/training/checkpoints/best2.pt   --output nepali/training/checkpoints/intrinsic_metrics.json --max-tokens 1000000
Evaluating validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [04:50<00:00,  4.68s/batch]
validation: loss=3.175546 ppl=23.9399 bpb=0.423113
Evaluating test: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [07:52<00:00,  7.61s/batch]
test: loss=3.156339 ppl=23.4845 bpb=0.417318

python generation_quality.py   --language hindi   --checkpoint hindi/training/checkpoints/best2.pt   --num-examples 100   --prefix-tokens 32   --continuation-tokens 64   --output-dir hindi/training/generation_quality
Generating greedy: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:49<00:00,  2.03example/s]
Generating sample_t0.5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:52<00:00,  1.13s/example]
Generating sample_t1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [04:43<00:00,  2.83s/example]
Generating sample_t1.5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:15<00:00,  1.96s/example]
Saved aggregate metrics: hindi/training/generation_quality/generation_metrics.json
Saved qualitative outputs: hindi/training/generation_quality/generations.jsonl
greedy: BLEU-4=4.56, chrF++=15.74, ROUGE-L=0.0251, repeat-3=0.5645, D-1=0.1393, D-2=0.2911
sample_t0.5: BLEU-4=4.71, chrF++=18.15, ROUGE-L=0.0444, repeat-3=0.3009, D-1=0.1863, D-2=0.4644
sample_t1: BLEU-4=3.69, chrF++=18.26, ROUGE-L=0.0150, repeat-3=0.0151, D-1=0.3409, D-2=0.8280
sample_t1.5: BLEU-4=1.83, chrF++=15.77, ROUGE-L=0.0104, repeat-3=0.0002, D-1=0.4852, D-2=0.9766

venv/bin/python generation_quality.py \
  --language nepali \
  --checkpoint nepali/training/checkpoints/best2.pt \
  --num-examples 100 \
  --prefix-tokens 32 \
  --continuation-tokens 64 \
  --output-dir nepali/training/generation_quality
Generating greedy: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:46<00:00,  2.17example/s]
Generating sample_t0.5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:48<00:00,  1.69s/example]
Generating sample_t1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [04:32<00:00,  2.73s/example]
Generating sample_t1.5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [05:18<00:00,  3.19s/example]
Saved aggregate metrics: nepali/training/generation_quality/generation_metrics.json
Saved qualitative outputs: nepali/training/generation_quality/generations.jsonl
greedy: BLEU-4=2.18, chrF++=17.58, ROUGE-L=0.0047, repeat-3=0.5795, D-1=0.1504, D-2=0.2960
sample_t0.5: BLEU-4=2.11, chrF++=19.49, ROUGE-L=0.0073, repeat-3=0.2801, D-1=0.2371, D-2=0.5190
sample_t1: BLEU-4=1.02, chrF++=19.10, ROUGE-L=0.0012, repeat-3=0.0190, D-1=0.3938, D-2=0.8588
sample_t1.5: BLEU-4=0.13, chrF++=16.30, ROUGE-L=0.0018, repeat-3=0.0010, D-1=0.5354, D-2=0.9714
