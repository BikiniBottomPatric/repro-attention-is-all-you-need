"""
诊断脚本 - 检查模型训练和推理是否正常
特别关注 cross-attention 是否正常工作
"""

import torch
import torch.nn.functional as F
import os
from config import get_config
from utils import load_model_and_tokenizers


def check_cross_attention(model, src_tensor, tgt_tensor, device):
    """检查 cross-attention 权重是否正常"""
    print("\n🔬 Cross-Attention 诊断:")
    
    model.eval()
    with torch.no_grad():
        # 获取 encoder 输出
        encoder_output, src_mask = model.encode(src_tensor)
        
        print(f"   Encoder输出: shape={encoder_output.shape}, mean={encoder_output.mean():.4f}, std={encoder_output.std():.4f}")
        print(f"   src_mask: shape={src_mask.shape}, True比例={src_mask.float().mean():.2%}")
        
        # 如果所有位置都被 mask 掉，这是严重问题！
        if src_mask.float().mean() < 0.1:
            print("   ⚠️ 警告: src_mask 中大部分位置被 mask 掉了！这会导致 cross-attention 失效！")
        
        # 手动执行 decoder 的一层，捕获 attention weights
        tgt_mask = model.create_decoder_mask(tgt_tensor)
        x = model.tgt_embedding(tgt_tensor)
        
        # 通过第一个 decoder 层
        layer = model.decoder_layers[0]
        
        # Self-attention
        self_attn_out, self_attn_weights = layer.self_attention(x, x, x, tgt_mask)
        x = layer.norm1(x + layer.dropout1(self_attn_out))
        
        # Cross-attention - 这是关键！
        cross_attn_out, cross_attn_weights = layer.cross_attention(x, encoder_output, encoder_output, src_mask)
        
        print(f"\n   Self-Attention 权重:")
        print(f"     shape: {self_attn_weights.shape}")
        print(f"     mean: {self_attn_weights.mean():.4f}")
        print(f"     max: {self_attn_weights.max():.4f}")
        
        print(f"\n   Cross-Attention 权重 (关键!):")
        print(f"     shape: {cross_attn_weights.shape}")
        print(f"     mean: {cross_attn_weights.mean():.4f}")
        print(f"     max: {cross_attn_weights.max():.4f}")
        print(f"     min: {cross_attn_weights.min():.4f}")
        
        # 检查 cross-attention 是否在真正关注源文
        # 如果权重非常均匀（接近 1/src_len），说明模型没有学会关注
        src_len = src_tensor.size(1)
        uniform_attn = 1.0 / src_len
        
        # 计算 attention 的"集中度"（熵）
        # 低熵 = 集中关注少数位置（好）
        # 高熵 = 均匀分布（坏，说明没学会）
        attn_entropy = -(cross_attn_weights * (cross_attn_weights + 1e-10).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(src_len)))
        
        print(f"\n   Attention 分析:")
        print(f"     源序列长度: {src_len}")
        print(f"     均匀分布期望值: {uniform_attn:.4f}")
        print(f"     实际熵: {attn_entropy:.4f}")
        print(f"     最大熵 (均匀分布): {max_entropy:.4f}")
        print(f"     熵比例: {attn_entropy/max_entropy:.2%}")
        
        if attn_entropy / max_entropy > 0.9:
            print("   ⚠️ Cross-attention 接近均匀分布！模型可能没有学会正确关注源文！")
        elif attn_entropy / max_entropy > 0.7:
            print("   ⚠️ Cross-attention 分布较为均匀，模型还在早期学习阶段")
        else:
            print("   ✅ Cross-attention 有一定的集中度，模型在学习关注源文")
        
        # 可视化第一个样本的 attention（文本形式）
        print(f"\n   第一个样本的 Cross-Attention (head 0, 第一个tgt位置关注src各位置):")
        attn_first = cross_attn_weights[0, 0, 0, :].cpu().numpy()  # (src_len,)
        for i, w in enumerate(attn_first[:min(10, len(attn_first))]):
            bar = "█" * int(w * 50)
            print(f"     src[{i}]: {w:.4f} {bar}")
        if len(attn_first) > 10:
            print(f"     ... (共 {len(attn_first)} 个位置)")


def diagnose_model(checkpoint_path, data_dir='./wmt14_data'):
    """诊断模型状态"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("🔍 模型诊断")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1️⃣ 加载模型...")
    model, src_tokenizer, tgt_tokenizer, vocab_info, device = load_model_and_tokenizers(
        checkpoint_path, data_dir, device
    )
    model.eval()
    print(f"   ✅ 模型加载成功")
    print(f"   设备: {device}")
    print(f"   词汇表大小: src={vocab_info.get('src_vocab_size', 'N/A')}, tgt={vocab_info.get('tgt_vocab_size', 'N/A')}")
    
    # 打印关键 token IDs
    print(f"\n   特殊 Token IDs:")
    print(f"     PAD: {vocab_info.get('pad_token_id')}")
    print(f"     BOS: {vocab_info.get('bos_token_id')}")
    print(f"     EOS: {vocab_info.get('eos_token_id')}")
    print(f"     src_pad: {vocab_info.get('src_pad_token_id')}")
    print(f"     tgt_pad: {vocab_info.get('tgt_pad_token_id')}")
    
    # 2. 检查权重统计
    print("\n2️⃣ 检查权重统计...")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            mean = param.data.mean().item()
            std = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            # 检查异常值
            has_nan = torch.isnan(param.data).any().item()
            has_inf = torch.isinf(param.data).any().item()
            
            if has_nan or has_inf or std < 1e-6 or std > 10:
                status = "⚠️ 异常"
            else:
                status = "✅"
            
            # 只打印关键层
            if 'attention' in name or 'embedding' in name or 'output_projection' in name:
                print(f"   {status} {name}: mean={mean:.4f}, std={std:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    
    # 3. 测试前向传播
    print("\n3️⃣ 测试前向传播...")
    test_text = "Hallo, wie geht es Ihnen?"  # 简单德语句子
    
    # 编码
    src_enc = src_tokenizer.encode(test_text)
    src_ids = src_enc.ids
    eos_id = src_tokenizer.token_to_id('</s>')
    bos_id = tgt_tokenizer.token_to_id('<s>')
    pad_id = tgt_tokenizer.token_to_id('<pad>')
    
    src_ids = src_ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    
    print(f"   源文本: {test_text}")
    print(f"   源tokens: {src_ids}")
    print(f"   源长度: {len(src_ids)}")
    
    # 检查 PAD token 是否正确
    print(f"\n   PAD Token 检查:")
    print(f"     pad_id = {pad_id}")
    print(f"     源序列中 PAD 数量: {sum(1 for t in src_ids if t == pad_id)}")
    print(f"     模型的 src_pad_id: {model.src_pad_id}")
    print(f"     模型的 tgt_pad_id: {model.tgt_pad_id}")
    
    if pad_id != model.src_pad_id:
        print(f"   ⚠️ 警告: 分词器 PAD ID ({pad_id}) 与模型 src_pad_id ({model.src_pad_id}) 不一致！")
    
    with torch.no_grad():
        # 编码器输出
        encoder_output, src_mask = model.encode(src_tensor)
        print(f"   Encoder输出形状: {encoder_output.shape}")
        print(f"   Encoder输出统计: mean={encoder_output.mean().item():.4f}, std={encoder_output.std().item():.4f}")
        
        # 检查encoder输出是否有效
        if encoder_output.std().item() < 1e-6:
            print("   ⚠️ Encoder输出方差过小，可能有问题！")
        
        # 解码器测试（单步）
        tgt_start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        decoder_output = model.decode(tgt_start, encoder_output, src_mask)
        logits = model.output_projection(decoder_output)
        
        print(f"   Decoder输出形状: {decoder_output.shape}")
        print(f"   Decoder输出统计: mean={decoder_output.mean().item():.4f}, std={decoder_output.std().item():.4f}")
        print(f"   Logits形状: {logits.shape}")
        
        # 查看top-5预测
        probs = torch.softmax(logits[0, 0], dim=-1)
        top_probs, top_ids = torch.topk(probs, 5)
        print(f"   Top-5预测:")
        for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
            token = tgt_tokenizer.decode([idx]) if hasattr(tgt_tokenizer, 'decode') else f"ID:{idx}"
            print(f"     {i+1}. '{token}' (prob={prob:.4f}, id={idx})")
    
    # 4. 检查 Cross-Attention（关键诊断！）
    print("\n4️⃣ Cross-Attention 诊断（关键！）...")
    tgt_start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    check_cross_attention(model, src_tensor, tgt_start, device)
    
    # 5. 使用greedy和beam分别生成
    print("\n5️⃣ 测试生成...")
    from inference import TransformerInference
    
    inference = TransformerInference(
        model, src_tokenizer, tgt_tokenizer, device,
        vocab_info['bos_token_id'], vocab_info['eos_token_id'], vocab_info['pad_token_id']
    )
    
    # Greedy
    print("   Greedy解码:")
    greedy_result = inference.greedy_decode(test_text, max_length=50)
    print(f"   输入: {test_text}")
    print(f"   输出: {greedy_result}")
    
    # Beam
    print("\n   Beam Search解码:")
    beam_result = inference.beam_search_decode(test_text, beam_size=4, max_length=50)
    print(f"   输入: {test_text}")
    print(f"   输出: {beam_result}")
    
    # 6. 测试更长的句子
    print("\n6️⃣ 测试WMT风格句子...")
    long_text = "Die Europäische Union hat neue Regeln für den Datenschutz eingeführt."
    
    greedy_long = inference.greedy_decode(long_text, max_length=100)
    print(f"   源文: {long_text}")
    print(f"   Greedy: {greedy_long}")
    
    beam_long = inference.beam_search_decode(long_text, beam_size=4, max_length=100)
    print(f"   Beam:   {beam_long}")
    
    print("\n" + "=" * 60)
    print("🔍 诊断完成")
    print("=" * 60)


def check_tokenizer_ids(data_dir='./wmt14_data'):
    """检查分词器的特殊 token IDs 是否正确"""
    print("\n" + "=" * 60)
    print("🔍 分词器 Token ID 检查")
    print("=" * 60)
    
    # 尝试加载 SentencePiece
    spm_model = os.path.join(data_dir, 'spm_shared.model')
    if os.path.exists(spm_model):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=spm_model)
            
            print("\nSentencePiece 分词器:")
            print(f"  词表大小: {sp.get_piece_size()}")
            print(f"\n  特殊 Token IDs (内置方法):")
            print(f"    sp.pad_id() = {sp.pad_id()}")
            print(f"    sp.unk_id() = {sp.unk_id()}")
            print(f"    sp.bos_id() = {sp.bos_id()}")
            print(f"    sp.eos_id() = {sp.eos_id()}")
            
            print(f"\n  piece_to_id 方法 (代码中使用的):")
            print(f"    piece_to_id('<pad>') = {sp.piece_to_id('<pad>')}")
            print(f"    piece_to_id('<unk>') = {sp.piece_to_id('<unk>')}")
            print(f"    piece_to_id('<s>') = {sp.piece_to_id('<s>')}")
            print(f"    piece_to_id('</s>') = {sp.piece_to_id('</s>')}")
            
            # 关键检查！
            if sp.pad_id() != sp.piece_to_id('<pad>'):
                print(f"\n  ⚠️ 严重警告: pad_id() != piece_to_id('<pad>')!")
                print(f"     这会导致 mask 计算错误，cross-attention 会失效！")
            
            # 测试编码
            test_text = "Hello world"
            encoded = sp.encode(test_text, out_type=int)
            print(f"\n  测试编码 '{test_text}':")
            print(f"    IDs: {encoded}")
            print(f"    解码: {sp.decode(encoded)}")
            
        except Exception as e:
            print(f"  加载失败: {e}")
    else:
        print(f"  SentencePiece 模型不存在: {spm_model}")


def check_data_alignment(data_dir='./wmt14_data', num_samples=5):
    """检查数据对齐是否正确"""
    print("\n" + "=" * 60)
    print("🔍 数据对齐检查")
    print("=" * 60)
    
    de_path = os.path.join(data_dir, 'valid.de')
    en_path = os.path.join(data_dir, 'valid.en')
    
    if not os.path.exists(de_path) or not os.path.exists(en_path):
        print(f"❌ 数据文件不存在: {data_dir}")
        return
    
    print(f"\n前{num_samples}个验证集样本:")
    with open(de_path, 'r', encoding='utf-8') as f_de, \
         open(en_path, 'r', encoding='utf-8') as f_en:
        for i, (de_line, en_line) in enumerate(zip(f_de, f_en)):
            if i >= num_samples:
                break
            print(f"\n样本 {i+1}:")
            print(f"  DE: {de_line.strip()[:100]}...")
            print(f"  EN: {en_line.strip()[:100]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='模型诊断')
    parser.add_argument('--checkpoint', default='/root/autodl-tmp/checkpoints/best_model.pt', 
                        help='检查点路径')
    parser.add_argument('--data-dir', default='./wmt14_data', help='数据目录')
    parser.add_argument('--check-data', action='store_true', help='检查数据对齐')
    
    args = parser.parse_args()
    
    # 首先检查分词器的 token IDs（这是最可能出问题的地方！）
    check_tokenizer_ids(args.data_dir)
    
    if args.check_data:
        check_data_alignment(args.data_dir)
    
    if os.path.exists(args.checkpoint):
        diagnose_model(args.checkpoint, args.data_dir)
    else:
        print(f"❌ 检查点不存在: {args.checkpoint}")
        print("请使用 --checkpoint 指定正确的路径")
