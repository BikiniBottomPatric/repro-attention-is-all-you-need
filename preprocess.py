#!/usr/bin/env python3
"""
预处理脚本 - 独立训练分词器（与训练完全解耦）

原论文使用 BPE (Byte Pair Encoding) 分词，词表大小约 37000。
本脚本支持 SentencePiece BPE（推荐，原论文使用）和 HuggingFace BPE。

使用方法：
    python preprocess.py                    # 使用config.py中的配置
    python preprocess.py --force            # 强制重新训练
    python preprocess.py --vocab-size 32000 # 自定义词表大小
"""

import os
import argparse


def train_sentencepiece(data_dir: str, vocab_size: int, model_type: str = 'bpe'):
    """训练 SentencePiece 分词器（原论文方式）"""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("需要安装 sentencepiece: pip install sentencepiece")
    
    train_de = os.path.join(data_dir, 'train.de')
    train_en = os.path.join(data_dir, 'train.en')
    
    if not os.path.exists(train_de) or not os.path.exists(train_en):
        raise FileNotFoundError(f"训练数据不存在: {train_de} 或 {train_en}")
    
    print(f"🔤 训练 SentencePiece 分词器")
    print(f"   算法: {model_type.upper()}")
    print(f"   词表大小: {vocab_size}")
    
    # 创建合并语料文件
    corpus_path = os.path.join(data_dir, 'spm_corpus.txt')
    print(f"   创建合并语料...")
    
    line_count = 0
    with open(corpus_path, 'w', encoding='utf-8') as f_out:
        with open(train_de, 'r', encoding='utf-8') as f_de:
            for line in f_de:
                text = line.strip().replace('\n', ' ')
                if text:
                    f_out.write(text + '\n')
                    line_count += 1
        with open(train_en, 'r', encoding='utf-8') as f_en:
            for line in f_en:
                text = line.strip().replace('\n', ' ')
                if text:
                    f_out.write(text + '\n')
                    line_count += 1
    
    print(f"   语料行数: {line_count:,}")
    print(f"   训练中... (约5-15分钟)")
    
    # 训练 SentencePiece
    spm_prefix = os.path.join(data_dir, 'spm_shared')
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=spm_prefix,
        vocab_size=vocab_size,
        model_type=model_type,  # bpe | unigram
        character_coverage=1.0,
        # 特殊token ID（与模型代码一致）
        pad_id=0,   # <pad>
        unk_id=1,   # <unk>
        bos_id=2,   # <s>
        eos_id=3,   # </s>
    )
    
    # 清理临时文件
    os.remove(corpus_path)
    
    print(f"✅ 分词器已保存:")
    print(f"   {spm_prefix}.model")
    print(f"   {spm_prefix}.vocab")
    
    # 验证
    sp = spm.SentencePieceProcessor(model_file=spm_prefix + '.model')
    print(f"   实际词表大小: {sp.get_piece_size()}")
    print(f"   特殊token: <pad>={sp.pad_id()}, <unk>={sp.unk_id()}, <s>={sp.bos_id()}, </s>={sp.eos_id()}")


def train_hf_bpe(data_dir: str, vocab_size: int):
    """训练 HuggingFace BPE 分词器（备选方式）"""
    try:
        from tokenizers import Tokenizer
        from tokenizers import decoders
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    except ImportError:
        raise ImportError("需要安装 tokenizers: pip install tokenizers")
    
    train_de = os.path.join(data_dir, 'train.de')
    train_en = os.path.join(data_dir, 'train.en')
    
    if not os.path.exists(train_de) or not os.path.exists(train_en):
        raise FileNotFoundError(f"训练数据不存在: {train_de} 或 {train_en}")
    
    print(f"🔤 训练 HuggingFace BPE 分词器")
    print(f"   词表大小: {vocab_size}")
    
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    
    # 创建分词器
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
    )
    
    # 迭代器
    def text_iterator():
        with open(train_de, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield line.strip()
        with open(train_en, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield line.strip()
    
    print("   训练中... (约5-15分钟)")
    tok.train_from_iterator(text_iterator(), trainer)
    tok.decoder = decoders.ByteLevel()
    
    # 保存
    save_path = os.path.join(data_dir, 'tokenizer_shared.json')
    tok.save(save_path)
    
    print(f"✅ 分词器已保存: {save_path}")
    print(f"   实际词表大小: {tok.get_vocab_size()}")


def main():
    # 从 config 读取默认值
    from config import get_config
    config = get_config()
    
    parser = argparse.ArgumentParser(description='预处理 - 训练分词器')
    parser.add_argument('--data-dir', default=config['data_dir'], 
                       help=f'数据目录 (默认: {config["data_dir"]})')
    parser.add_argument('--backend', default=config.get('tokenizer_backend', 'sentencepiece'),
                       choices=['sentencepiece', 'bpe'],
                       help='分词器后端')
    parser.add_argument('--vocab-size', type=int, 
                       default=config.get('sp_vocab_size', config.get('vocab_size', 37000)),
                       help='词表大小')
    parser.add_argument('--model-type', default=config.get('sp_model_type', 'bpe'),
                       choices=['bpe', 'unigram'],
                       help='SentencePiece算法类型 (默认: bpe，原论文使用)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新训练（删除已有分词器）')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("📦 Transformer 预处理 - 分词器训练")
    print("=" * 50)
    print(f"数据目录: {args.data_dir}")
    print(f"分词器后端: {args.backend}")
    print(f"词表大小: {args.vocab_size}")
    if args.backend == 'sentencepiece':
        print(f"算法类型: {args.model_type}")
    print()
    
    # 检查数据文件
    train_de = os.path.join(args.data_dir, 'train.de')
    train_en = os.path.join(args.data_dir, 'train.en')
    
    if not os.path.exists(train_de) or not os.path.exists(train_en):
        print(f"❌ 训练数据不存在!")
        print(f"   需要: {train_de}")
        print(f"   需要: {train_en}")
        return
    
    # 检查是否已存在
    spm_model = os.path.join(args.data_dir, 'spm_shared.model')
    bpe_json = os.path.join(args.data_dir, 'tokenizer_shared.json')
    
    if args.force:
        # 强制删除已有分词器
        for f in [spm_model, spm_model.replace('.model', '.vocab'), bpe_json]:
            if os.path.exists(f):
                os.remove(f)
                print(f"🗑️ 已删除: {f}")
    else:
        # 检查是否已存在
        if args.backend == 'sentencepiece' and os.path.exists(spm_model):
            print(f"✅ 分词器已存在: {spm_model}")
            print("   使用 --force 强制重新训练")
            return
        if args.backend == 'bpe' and os.path.exists(bpe_json):
            print(f"✅ 分词器已存在: {bpe_json}")
            print("   使用 --force 强制重新训练")
            return
    
    # 训练分词器
    if args.backend == 'sentencepiece':
        train_sentencepiece(args.data_dir, args.vocab_size, args.model_type)
    else:
        train_hf_bpe(args.data_dir, args.vocab_size)
    
    print()
    print("🎉 预处理完成！")
    print("   下一步: python train.py")


if __name__ == '__main__':
    main()
