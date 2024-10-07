echo "Preparing streaming dataset..."
python prepare_streaming_dataset.py \
    --path TranCongThien/viet_wiki_data \
    --out_root vi-wiki \
    --tokenizer initial-vi-gemma-2b