from transformers import BertConfig, TrainingArguments, Trainer
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser

from models.model import Caser
from models.metric import compute_metric

from utils.preprocess import create_datasets, tuncate_and_padding

@dataclass
class DataArguments:
    max_seq_length: int=field(
        default=50, metadata={"help": "모델에 입력될 최대 시퀀스 길이"}
    )
    split: str=field(
        default="leave_one_out", metadata={"help": "데이터 스플릿 방식"} # 
    )
    input_files: str=field(
        default=None, metadata={"help": "데이터 원천 파일"}
    )
    min_rating: int=field(
        default=4, metadata={"help": "최소 평점"}
    )
    min_uc: int=field(
        default=10, metadata={"help": "유지할 유저의 최소 리뷰 수"}
    )
    min_sc: int=field(
        default=0, metadata={"help": "유지할 아이템의 최소 리뷰 수"}
    )
    
@dataclass
class ModelArguments:
    d: int=field(
        default=64
    )
    nv: int=field(
        default=4
    )
    nh: int=field(
        default=16,
    )
    drop: float=field(
        default=0.5
    )
    ac_conv: str=field(
        default="relu"
    )
    ac_fc: str=field(
        default="relu"
    )
    

if __name__ == "__main__":
    
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()
    
    
    train_dataset, val_dataset, test_dataset, smap = create_datasets(data_args.input_files, data_args.min_rating, 
                                                               data_args.min_sc, data_args.min_uc, data_args.split)
    
    
    train_dataset = tuncate_and_padding(train_dataset, True, data_args.max_seq_length)
    val_dataset = tuncate_and_padding(val_dataset, False, data_args.max_seq_length)
    
    num_items = len(smap) + 2
    num_users = len(train_dataset)
    
    model_args.L = data_args.max_seq_length
    
    
    model = Caser(num_users, num_items, model_args)
    
    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, 
                      eval_dataset=val_dataset,
                      compute_metrics=compute_metric)
    
    
    trainer.train()
    #option.