echo "Download dataset"
python train_tokenizer.py download_wikidata --data_path TranCongThien/viet_wiki_data --saved_path vi_clean_corpus.txt

echo "Train a new Tokenizer"
python train_tokenizer.py train_tokenizer --sp_model_name vi-tokenizer-10k

echo "Merge new tokenizer to gemma-Tokenizer"
python train_tokenizer.py merge_tokenizer \
    --source_tokenizer_dir google/gemma-2b \
    --new_tokenizer_model vi-tokenizer-10k \
    --new_tokenizer_dir initial-vi-gemma-2b

echo "Initialize new model, this process may take some time to complete..."
python train_tokenizer.py reinit_model \
    --model_name google/gemma-2b \
    --new_tokenizer_dir initial-vi-gemma-2b

