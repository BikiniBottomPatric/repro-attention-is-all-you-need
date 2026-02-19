"""
数据处理 - 使用 HuggingFace datasets 的简洁管线
加载WMT14、加载分词器、批采样与动态裁剪

注意：分词器训练已解耦到 preprocess.py，请先运行 preprocess.py 再运行 train.py
"""

import os
import random
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer
from tokenizers import decoders

# SentencePiece（可选）
try:
    import sentencepiece as spm  # type: ignore
    HAS_SPM = True
except Exception:
    HAS_SPM = False

# 导入共享的 SPWrapper 类
from utils import SPWrapper
 

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️ 建议安装 datasets: pip install datasets")


class Collator:
    """批处理函数 - 动态按batch实际长度裁剪，避免固定padding引发OOM"""

    def __init__(self, src_pad_token_id: int, tgt_pad_token_id: int):
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 先按batch内最大长度进行padding，再进行stack
        src_list = [item['src_ids'] if isinstance(item['src_ids'], torch.Tensor) else torch.tensor(item['src_ids'], dtype=torch.long)
                    for item in batch]
        tgt_list = [item['tgt_ids'] if isinstance(item['tgt_ids'], torch.Tensor) else torch.tensor(item['tgt_ids'], dtype=torch.long)
                    for item in batch]

        src_batch = pad_sequence(src_list, batch_first=True, padding_value=self.src_pad_token_id)  # (B, S_max)
        tgt_batch = pad_sequence(tgt_list, batch_first=True, padding_value=self.tgt_pad_token_id)  # (B, T_max)

        # 动态裁剪到本batch内的最大非pad长度
        with torch.no_grad():
            # 源侧最大有效长度
            src_valid = (src_batch != self.src_pad_token_id)
            if src_valid.any():
                src_lengths = src_valid.sum(dim=1)
                src_max_len = int(src_lengths.max().item())
            else:
                src_max_len = src_batch.size(1)

            # 目标侧最大有效长度
            tgt_valid = (tgt_batch != self.tgt_pad_token_id)
            if tgt_valid.any():
                tgt_lengths = tgt_valid.sum(dim=1)
                tgt_max_len = int(tgt_lengths.max().item())
            else:
                tgt_max_len = tgt_batch.size(1)

        # 切到最小必要长度（至少1，避免空张量）
        src_max_len = max(1, src_max_len)
        tgt_max_len = max(2, tgt_max_len)  # 目标至少保留2以便后续shift

        src_batch = src_batch[:, :src_max_len].contiguous()
        tgt_batch = tgt_batch[:, :tgt_max_len].contiguous()

        return {
            'src_ids': src_batch,
            'tgt_ids': tgt_batch,
        }


class TokenBatchSampler:
    """简洁的按 token 数打包采样器（单次长度计算 + 简单排序）。"""

    def __init__(
        self,
        dataset: Dataset,
        max_tokens_per_batch: int,
        max_sentences_per_batch: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.max_tokens = int(max_tokens_per_batch)
        self.max_sentences = int(max_sentences_per_batch) if max_sentences_per_batch else None
        self.shuffle = shuffle
        
        # 优化：优先使用预计算的 'length' 列，避免逐行读取导致的性能瓶颈
        if hasattr(dataset, 'column_names') and 'length' in dataset.column_names:
            # 如果存在 length 列，直接读取（非常快）
            self.lengths = dataset['length']
            if isinstance(self.lengths, torch.Tensor):
                self.lengths = self.lengths.tolist()
        else:
            # 回退到旧方法（较慢）
            print("⚠️ 警告: 数据集缺少 'length' 列，正在逐行计算长度（可能较慢）...")
            def item_len(i: int) -> int:
                ex = dataset[i]
                src_len = int(len(ex['src_ids']))
                tgt_len = int(len(ex['tgt_ids']))
                return max(1, max(src_len, tgt_len))
            self.lengths: List[int] = [item_len(i) for i in range(len(dataset))]
            
        self._build_order()

    def _build_order(self) -> None:
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)
        # 简洁：直接按长度排序，减少padding
        self.ordered = sorted(idxs, key=self.lengths.__getitem__)

    def __iter__(self):
        if self.shuffle:
            self._build_order()
        
        # 1. 先生成所有 batch
        batches = []
        batch: List[int] = []
        max_len = 0
        
        for i in self.ordered:
            l = self.lengths[i]
            new_max = l if l > max_len else max_len
            new_cnt = len(batch) + 1
            if (new_max * new_cnt <= self.max_tokens) and (self.max_sentences is None or new_cnt <= self.max_sentences):
                batch.append(i)
                max_len = new_max
            else:
                if batch:
                    batches.append(batch)
                batch = [i]
                max_len = l
        if batch:
            batches.append(batch)
            
        # 2. 关键修复：打乱 batch 的顺序！
        # 之前的代码虽然打乱了样本，但是 sorted(key=len) 后又变回了按长度严格排序
        # 导致每个 epoch 都是从短句子训练到长句子 (Curriculum Learning)，这会严重阻碍收敛
        if self.shuffle:
            random.shuffle(batches)
            # 同时打乱每个 batch 内的样本顺序（更彻底的随机化）
            for batch in batches:
                random.shuffle(batch)
            
        # 3. Yield batches
        yield from batches

    def __len__(self) -> int:
        # 粗略估计：平均长度估算批次数
        avg = sum(self.lengths) / max(1, len(self.lengths))
        per = max(1, int(self.max_tokens // max(1, avg)))
        return (len(self.lengths) + per - 1) // per


def create_data_loaders(config: Dict[str, Any]):
    """创建数据加载器
    - 当 use_hf_data=True 时：使用 datasets.map 动态管线（无需 *.pt 预处理文件）
    - 否则：沿用已有 TensorDataset 缓存文件
    """
    data_dir = config['data_dir']

    # 仅支持 HuggingFace datasets 管线（与社区实践一致）
    if config.get('use_hf_data', False):
        if not HAS_DATASETS:
            raise ImportError("需要安装 datasets: pip install datasets")

        from datasets import Dataset
        
        # 直接从本地文本加载（不尝试连接HuggingFace Hub）
        def _read_parallel(split: str):
            de_path = os.path.join(data_dir, f"{split}.de")
            en_path = os.path.join(data_dir, f"{split}.en")
            if not (os.path.exists(de_path) and os.path.exists(en_path)):
                raise FileNotFoundError(
                    f"未找到本地 {split}.de/.en，请确保数据文件存在于 {data_dir}"
                )
            de_list: List[str] = []
            en_list: List[str] = []
            with open(de_path, 'r', encoding='utf-8') as f_de, open(en_path, 'r', encoding='utf-8') as f_en:
                for de_line, en_line in zip(f_de, f_en):
                    de_txt = de_line.strip()
                    en_txt = en_line.strip()
                    if de_txt and en_txt:
                        de_list.append(de_txt)
                        en_list.append(en_txt)
            if len(de_list) == 0:
                raise ValueError(f"本地 {split} 文本为空")
            print(f"📂 从本地加载 {split}: {len(de_list)} 句对")
            return Dataset.from_dict({'de': de_list, 'en': en_list})
        
        train_raw = _read_parallel('train')
        valid_raw = _read_parallel('valid')

        # 加载分词器（分词器训练已解耦到 preprocess.py）
        special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        tok_backend = config.get('tokenizer_backend', 'sentencepiece')  # sentencepiece | bpe

        def _get_text(ex, key):
            if 'translation' in ex:
                return ex['translation'][key]
            return ex[key]

        # 根据后端加载分词器
        if tok_backend == 'sentencepiece':
            if not HAS_SPM:
                raise ImportError("需要安装 sentencepiece: pip install sentencepiece")
            spm_model = os.path.join(data_dir, 'spm_shared.model')
            if not os.path.exists(spm_model):
                raise FileNotFoundError(
                    f"❌ 分词器不存在: {spm_model}\n"
                    f"   请先运行预处理: python preprocess.py"
                )
            print(f"📂 加载 SentencePiece 分词器: {spm_model}")
            sp = spm.SentencePieceProcessor(model_file=spm_model)
            src_tokenizer = SPWrapper(sp)
            tgt_tokenizer = src_tokenizer
        else:  # bpe (HuggingFace)
            shared_path = os.path.join(data_dir, 'tokenizer_shared.json')
            if not os.path.exists(shared_path):
                raise FileNotFoundError(
                    f"❌ 分词器不存在: {shared_path}\n"
                    f"   请先运行预处理: python preprocess.py --backend bpe"
                )
            print(f"📂 加载 HuggingFace BPE 分词器: {shared_path}")
            src_tokenizer = Tokenizer.from_file(shared_path)
            # 确保解码器存在
            try:
                if getattr(src_tokenizer, 'decoder', None) is None:
                    src_tokenizer.decoder = decoders.ByteLevel()
            except Exception:
                pass
            tgt_tokenizer = src_tokenizer
        
        # 校验特殊token
        missing = [t for t in special_tokens if src_tokenizer.token_to_id(t) is None]
        if missing:
            raise ValueError(f"分词器缺少特殊token: {missing}")

        # ID 常量
        src_pad = src_tokenizer.token_to_id('<pad>')
        tgt_pad = tgt_tokenizer.token_to_id('<pad>')
        bos_id = tgt_tokenizer.token_to_id('<s>')
        eos_id = tgt_tokenizer.token_to_id('</s>')
        unk_id = tgt_tokenizer.token_to_id('<unk>')

        max_src_len = int(config['max_src_len'])
        max_tgt_len = int(config['max_tgt_len'])

        num_proc = int(config.get('num_workers', 8))
        num_proc = max(1, num_proc // 2)  # map 不宜开太多进程

        def _encode_example(ex):
            de_txt = _get_text(ex, 'de')
            en_txt = _get_text(ex, 'en')
            if not de_txt or not en_txt:
                return {'src_ids': [], 'tgt_ids': [], 'length': 0}
            
            # 源序列编码：加上 EOS（论文标准做法）
            # Source: [word1, word2, ..., EOS]
            src_core = src_tokenizer.encode(de_txt).ids
            if config.get('drop_too_long', True) and len(src_core) + 1 > max_src_len:  # +1 for EOS
                return {'src_ids': [], 'tgt_ids': []}
            src_core = src_core[:max_src_len - 1]  # 留一个位置给 EOS
            src_ids = src_core + [eos_id]  # ✅ 源序列加上 EOS
            
            # 目标序列编码：BOS + content + EOS（论文标准做法）
            # Target: [BOS, word1, word2, ..., EOS]
            inner_max = max(0, max_tgt_len - 2)  # 留位置给 BOS 和 EOS
            tgt_core = tgt_tokenizer.encode(en_txt).ids
            
            # 句对长度过滤（常见实践）：过度不匹配的句对可丢弃
            if config.get('drop_too_long', True):
                src_len = len(src_core) if src_core else 1
                tgt_len = len(tgt_core) if tgt_core else 1
                ratio = max(src_len / tgt_len, tgt_len / src_len)
                if ratio > float(config.get('length_ratio_threshold', 2.0)):
                    return {'src_ids': [], 'tgt_ids': []}
            
            tgt_core = tgt_core[:inner_max]
            tgt_ids = [bos_id] + tgt_core + [eos_id]
            return {'src_ids': src_ids, 'tgt_ids': tgt_ids}

        print("🔄 对训练/验证集进行分词映射（datasets.map）…")
        # 恢复默认缓存机制（load_from_cache_file=None/True）
        # 只要之前的处理参数没变，datasets会自动加载已有的缓存，不会重新跑7个小时
        train_enc = train_raw.map(_encode_example, remove_columns=train_raw.column_names, num_proc=num_proc)
        valid_enc = valid_raw.map(_encode_example, remove_columns=valid_raw.column_names, num_proc=num_proc)

        # 过滤空或过长被丢弃的样本
        train_enc = train_enc.filter(lambda ex: len(ex['src_ids']) > 0 and len(ex['tgt_ids']) > 1)
        valid_enc = valid_enc.filter(lambda ex: len(ex['src_ids']) > 0 and len(ex['tgt_ids']) > 1)

        # 转为torch格式 (移除 length 列要求，匹配旧缓存)
        train_enc.set_format(type='torch', columns=['src_ids', 'tgt_ids'])
        valid_enc.set_format(type='torch', columns=['src_ids', 'tgt_ids'])

        # Collator：保留现有实现，动态按batch裁剪
        collator = Collator(src_pad_token_id=src_pad, tgt_pad_token_id=tgt_pad)

        # 构建批采样器（长度基于实际序列长度；此处不依赖pad）
        max_tokens = config['max_tokens_per_batch']
        max_sentences = config.get('max_sentences_per_batch')

        train_sampler = TokenBatchSampler(
            train_enc, max_tokens_per_batch=max_tokens, max_sentences_per_batch=max_sentences,
            shuffle=True
        )
        val_sampler = TokenBatchSampler(
            valid_enc, max_tokens_per_batch=max_tokens, max_sentences_per_batch=max_sentences,
            shuffle=False
        )

        num_workers = config.get('num_workers', 8)
        train_loader = DataLoader(
            train_enc, batch_sampler=train_sampler, collate_fn=collator,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            valid_enc, batch_sampler=val_sampler, collate_fn=collator,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )

        vocab_info = {
            'src_vocab_size': src_tokenizer.get_vocab_size(),
            'tgt_vocab_size': tgt_tokenizer.get_vocab_size(),
            'pad_token_id': tgt_pad,
            'src_pad_token_id': src_pad,
            'tgt_pad_token_id': tgt_pad,
            'bos_token_id': bos_id,
            'eos_token_id': eos_id,
            'unk_token_id': unk_id,
            # 共享词表时 src 的 BOS/EOS 与 tgt 相同，用于评估回退
            'src_bos_token_id': src_tokenizer.token_to_id('<s>'),
            'src_eos_token_id': src_tokenizer.token_to_id('</s>'),
        }

        print("✅ DataLoader创建完成 (datasets 管线)")
        print(f"   训练: {len(train_enc):,} 样本")
        print(f"   验证: {len(valid_enc):,} 样本")
        print(f"   词汇: src={vocab_info['src_vocab_size']}, tgt={vocab_info['tgt_vocab_size']}")

        return train_loader, val_loader, vocab_info

    # 不再支持旧缓存路径/离线预处理
    raise FileNotFoundError("仅支持 datasets 管线：请在 config 中设置 use_hf_data=True")
