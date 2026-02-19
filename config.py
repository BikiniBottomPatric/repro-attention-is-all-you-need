"""
配置文件 - 简洁清晰的参数配置
提供三种预设配置，参数含义明确
"""

# 基础配置 - 严格按照原论文"Attention Is All You Need"设置
# 针对 RTX Pro 6000 96GB 优化
CONFIG = {
    # 数据配置 - 严格按原论文5.1节: ~25K source + 25K target ≈ 50K tokens/batch
    # 纯相对路径：假设代码和数据在同一目录下
    # 'data_dir': './wmt14_data',  
    'data_dir': './multi30k_data', # 本地验证用小型数据集
    
    # 批大小解释：
    # 论文定义: "Each batch contained ~25,000 source tokens and ~25,000 target tokens"
    # 你的 TokenBatchSampler 已经限制单批 ~25k tokens，符合论文设置
    # 梯度累积可选：accumulate=1 已匹配论文单批大小
    # 如果训练不稳定，可以尝试 accumulate=2 或 4 增加有效 batch
    # 'max_tokens_per_batch': 25000,
    'max_tokens_per_batch': 4000,  # 96GB VRAM 优化 (FP32)
    'max_sentences_per_batch': None,  # 可选：每批最大句子数上限（保险阈值）
    'accumulate_grad_batches': 6,   # 无需累积，物理 Batch 直接对齐论文标准
    
    'max_src_len': 256,  # 合理的序列长度，平衡内存和性能
    'max_tgt_len': 256,  # 合理的序列长度，平衡内存和性能
    
    # 模型配置 - 严格按论文Base模型 (Table 3)
    'd_model': 512,      # 论文Table 3: Base model
    'num_heads': 8,      # 论文Table 3: h=8
    'num_encoder_layers': 6,  # 论文Table 3: N=6 
    'num_decoder_layers': 6,
    'd_ff': 2048,        # 论文Table 3: d_ff=2048
    'dropout': 0.1,      # 论文5.4节: P_drop=0.1
    
    # 训练配置 - 严格按论文设置
    'warmup_steps': 4000,     # 论文5.3节: warmup_steps=4000
    'lr_scale': 1.0,          # 论文公式隐含值为1.0 (虽然社区常用2.0加速，但为了严格复现改回1.0)
    'label_smoothing': 0.1,   # 论文5.4节: ε_ls=0.1
    'num_epochs': 30,         # Multi30k 验证用 30 轮即可
    # 'num_epochs': 45,         # 原论文100K步: 4.5M×25tokens / 50K batch ≈ 2250 steps/epoch, 需~45 epochs
    # 注意：如果BatchSize减半了，Step数会翻倍，这里Epoch数保持不变，总Step数会自动翻倍匹配
    
    # 验证策略：完整验证集（提供稳定准确的指标）
    # 验证只在每个epoch结束时进行，使用完整验证集
    # 这样能避免部分数据导致的波动，更真实反映模型收敛情况
    
    # 保存配置
    # 改为相对路径，适应不同环境
    'save_dir': './checkpoints_multi30k',
    'save_interval': 1,  # 每个epoch保存一次（best_model.pt始终保存）
    
    # 评估配置（最终BLEU评估使用）
    'eval_batch_size': 32,     # 推理批处理大小（evaluate.py使用）
    'eval_method': 'greedy',   # 训练时用greedy评估（更快，便于诊断）
    'eval_beam_size': 4,       # 束搜索大小 (tensor2tensor官方实现依据)
    'eval_max_length': 100,    # 恢复正常最大长度
    'eval_length_penalty': 0.6,   # 恢复论文设置
    'eval_bleu_per_epoch': True,   # 是否在每个epoch结束时进行BLEU评估
    'eval_bleu_every_n_epochs': 1, # 每多少个epoch计算一次BLEU（1=每个epoch）
    # 推理配置 - 严格按原论文（纯 beam search + length penalty，无额外 trick）
    # 'no_repeat_ngram_size': 0,     # ❌ 关闭 n-gram 重复抑制（原论文无此设置）
    # 'min_decode_length': 0,        # ❌ 关闭最小生成长度（原论文无此设置）
    # 'eos_bias': 0.0,               # ❌ 关闭 EOS 偏置（原论文无此设置）
    # 'repetition_penalty': 1.0,     # ❌ 关闭重复惩罚（原论文无此设置）
    
    # 词汇配置
    'vocab_size': 10000,       # 论文标准词汇大小
    # 'vocab_size': 10000,       # Multi30k 小型词表
    'vocab_mode': 'shared',    # 默认共享词表，更贴近论文与社区常见实践
    
    # 性能优化配置 - 针对 RTX Pro 6000 + 22核CPU 优化
    'num_workers': 16,         # 释放 CPU 性能加速数据加载
    # 'use_compile': False,      # 暂时关闭 - torch.compile与mask操作有兼容性问题
    # 'use_amp': False,          # 关闭混合精度训练以还原论文设置，避免数值不稳定性
    # 'compile_mode': 'reduce-overhead',  # 平衡编译时间和运行效率
    'use_hf_data': True,       # 使用datasets.map动态数据管线（无需*.pt缓存）
    # 数据集选择（可切换到 iwslt2014 等）
    'hf_dataset': 'wmt14',     # HF 数据集名称: wmt14 / iwslt2014 / ...
    'hf_subset': 'de-en',      # 数据集子配置
    
    # 数据清洗（与社区实践一致）
    'drop_too_long': True,           # 训练集中丢弃过长样本（而非仅截断）
    'length_ratio_threshold': 2.0,   # 放松长度比阈值，保留更多样本

    # 分词器后端：
    # 'bpe' = HuggingFace Tokenizers ByteLevel BPE（GPT-2/RoBERTa风格）
    # 'sentencepiece' = Google SentencePiece（原论文tensor2tensor使用，推荐还原）
    'tokenizer_backend': 'sentencepiece',
    # 如需强制重训分词器（清理旧文件后重训），设置为 True
    'tokenizer_force_retrain': True,  # 切换到WMT14需要重新训练分词器
    'sp_vocab_size': 10000,
    'sp_model_type': 'bpe',  # bpe | unigram（原论文使用BPE）
    'sp_character_coverage': 1.0,
}

def get_config():
    """获取配置"""
    return CONFIG.copy()


def print_config():
    """打印配置详情"""
    config = get_config()
    
    print(f"\n📋 TRANSFORMER 配置:")
    print("=" * 40)
    
    print("📁 数据:")
    print(f"  数据目录: {config['data_dir']}")
    print(f"  每批最大tokens: {config['max_tokens_per_batch']}")
    if config.get('max_sentences_per_batch'):
        print(f"  每批最大句子数: {config['max_sentences_per_batch']}")
    print(f"  梯度累积: {config['accumulate_grad_batches']} 步")
    # 修正显示：有效tokens是 accumulated * max_tokens (这里近似为 src+tgt)
    # 原先的计算有误导性，现在明确为 Step Batch Size
    # 注意：max_tokens_per_batch 限制的是 max(src, tgt)，所以单批实际包含 src+tgt ~ 2*max_tokens (如果填满)
    # 但原论文的 25000 指的是 src 和 tgt 各 25000。
    # 我们的 sampler 限制 max(src, tgt) * batch <= 25000
    # 所以单批 src <= 25000, tgt <= 25000. 
    # 正好对应论文的一个 Batch。
    effective_tokens = "25k Src + 25k Tgt" if config['accumulate_grad_batches'] == 1 else f"{25 * config['accumulate_grad_batches']}k Src + {25 * config['accumulate_grad_batches']}k Tgt"
    
    print(f"  有效批大小: {effective_tokens} (匹配原论文: 25k+25k)" if config['accumulate_grad_batches'] == 1 else f"  有效批大小: {effective_tokens} (⚠️ 大于原论文)")
    print(f"  最大源长度: {config['max_src_len']}")
    print(f"  最大目标长度: {config['max_tgt_len']}")
    
    print("\n🏗️ 模型:")
    print(f"  模型维度: {config['d_model']}")
    print(f"  注意力头数: {config['num_heads']}")
    print(f"  Encoder层数: {config['num_encoder_layers']}")
    print(f"  Decoder层数: {config['num_decoder_layers']}")
    print(f"  FFN维度: {config['d_ff']}")
    print(f"  Dropout: {config['dropout']}")
    
    print("\n🚀 训练:")
    print(f"  Warmup步数: {config['warmup_steps']}")
    print(f"  学习率缩放: {config.get('lr_scale', 1.0)}")
    print(f"  标签平滑: {config['label_smoothing']}")
    print(f"  训练轮数: {config['num_epochs']}")
    print(f"  保存间隔: {config['save_interval']} epochs")
    print(f"  保存目录: {config['save_dir']}")
    print(f"  验证策略: 完整验证集 (每epoch结束时进行)")
    
    print("\n📊 评估:")
    print(f"  推理批处理大小: {config['eval_batch_size']}")
    print(f"  解码方法: {config['eval_method']}")
    print(f"  束搜索大小: {config['eval_beam_size']}")
    print(f"  最大生成长度: {config['eval_max_length']}")
    print(f"  每epoch BLEU评估: {'✅' if config['eval_bleu_per_epoch'] else '❌'}")
    
    print("\n⚡ 性能优化:")
    print(f"  DataLoader进程数: {config['num_workers']}")
    # print(f"  torch.compile: {'✅' if config['use_compile'] else '❌'} ({config.get('compile_mode', 'N/A')}模式)")
    # print(f"  混合精度训练: {'✅' if config['use_amp'] else '❌'} (节省显存)")
    print(f"  词汇表大小: {config['vocab_size']}")
    
    # 预估参数量
    params = estimate_params(config)
    print(f"\n🔧 预估:")
    print(f"  参数数量: ~{params:,}")
    fp32_size = params * 4 / 1024 / 1024
    fp16_size = params * 2 / 1024 / 1024
    print(f"  模型大小(FP32): ~{fp32_size:.1f} MB")
    print(f"  模型大小(FP16): ~{fp16_size:.1f} MB (AMP启用时)")
    print(f"  预期加速: 2-3x (torch.compile + AMP)")


def estimate_params(config):
    """预估模型参数数量"""
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_encoder_layers'] + config['num_decoder_layers']
    vocab_size = config.get('vocab_size', 37000)  # 从config读取，避免硬编码
    
    # 参数计算常量
    SRC_TGT_EMBEDDINGS = 2  # src + tgt 两个embedding层
    ATTN_PROJECTIONS = 4    # Q, K, V, O 四个注意力投影层
    NORMS_PER_LAYER = 3     # 每层平均3个LayerNorm (encoder:2, decoder:3)
    
    # 嵌入层参数
    embedding_params = vocab_size * d_model * SRC_TGT_EMBEDDINGS
    
    # 注意力层参数  
    attn_params_per_layer = ATTN_PROJECTIONS * d_model * d_model
    
    # FFN参数
    ffn_params_per_layer = d_model * d_ff + d_ff * d_model
    
    # Layer Norm参数
    norm_params_per_layer = d_model * NORMS_PER_LAYER
    
    # 总参数
    total_params = (embedding_params + 
                   num_layers * (attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer) +
                   d_model * vocab_size)  # 输出投影
    return total_params

if __name__ == "__main__":
    print_config()
