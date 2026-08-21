python3 ./scripts/train_bpe.py --input ../datasets/splits/train.jsonl --model-prefix unigram_4000 --vocab-size 4000
python3 ./scripts/train_bpe.py --input ../datasets/splits/train.jsonl --model-prefix unigram_6000 --vocab-size 6000
python3 ./scripts/train_bpe.py --input ../datasets/splits/train.jsonl --model-prefix unigram_8000 --vocab-size 8000
python3 ./scripts/train_bpe.py --input ../datasets/splits/train.jsonl --model-prefix unigram_10000 --vocab-size 10000
python3 ./scripts/train_bpe.py --input ../datasets/splits/train.jsonl --model-prefix unigram_12000 --vocab-size 12000
python3 ./scripts/train_bpe.py --input ../datasets/splits/train.jsonl --model-prefix unigram_16000 --vocab-size 16000
