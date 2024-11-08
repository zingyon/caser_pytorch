# caser_pytorch

### How to run

```sh
python run.py --input_files ./data/10core/ml-1m.text --output_dir ./output --do_train --do_eval --eval_strategy epoch --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 256 --num_train_epochs 10 --logging_steps 10
```
