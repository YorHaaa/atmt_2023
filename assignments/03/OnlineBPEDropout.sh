subword-nmt apply-bpe -c ./data/en-fr/bpe/bpe.codes --vocabulary ./data/en-fr/bpe/train_dict.en  < ./data/en-fr/preprocessed/train.en > ./data/en-fr/bpe/train.en --dropout 0.1
subword-nmt apply-bpe -c ./data/en-fr/bpe/bpe.codes --vocabulary ./data/en-fr/bpe/train_dict.fr  < ./data/en-fr/preprocessed/train.fr > ./data/en-fr/bpe/train.fr --dropout 0.1
python preprocess.py --target-lang en --source-lang fr --dest-dir ./data/en-fr/prepared/ --train-prefix ./data/en-fr/bpe/train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000