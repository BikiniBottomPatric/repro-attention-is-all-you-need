#!/usr/bin/env python3
"""
独立评估脚本 - 专门用于计算BLEU分数和其他评估指标
与训练过程完全解耦，可以单独运行

配置管理:
    - 默认参数从 config.py 加载
    - 命令行参数可以覆盖默认值

使用方法:
    python evaluate.py --model checkpoints/best_model.pt
    python evaluate.py --model checkpoints/best_model.pt --method beam --batch_size 8
"""

import os
import torch
import sacrebleu
import argparse
# 可选的进度条支持
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
from inference import TransformerInference
from config import get_config
from utils import load_model_and_tokenizers  # 使用共享函数避免重复

 

def evaluate_bleu(model, src_tokenizer, tgt_tokenizer, vocab_info, device, 
                 data_dir, batch_size=32, method='greedy', beam_size=4, max_length=256, length_penalty=0.6, split='test'):
    """在指定拆分上评估BLEU分数 (split: 'valid' | 'test')"""
    split_name = '验证集' if split == 'valid' else '测试集'
    print(f"BLEU评估: {split_name}, method={method}, bs={batch_size}")
    if method == 'beam':
        print(f"beam={beam_size}, alpha={length_penalty}")
    
    # 创建推理器
    inference = TransformerInference(
        model, src_tokenizer, tgt_tokenizer, device,
        vocab_info['bos_token_id'], vocab_info['eos_token_id'], vocab_info['pad_token_id']
    )
    
    # 加载原始文本以作参考（优先）
    src_texts = []
    ref_texts = []
    src_txt = os.path.join(data_dir, f'{split}.de')
    tgt_txt = os.path.join(data_dir, f'{split}.en')
    use_raw_refs = os.path.exists(src_txt) and os.path.exists(tgt_txt)

    if use_raw_refs:
        print("读取原始参考...")
        with open(src_txt, 'r', encoding='utf-8') as f_src, open(tgt_txt, 'r', encoding='utf-8') as f_tgt:
            for s, r in zip(f_src, f_tgt):
                s_stripped = s.strip()
                r_stripped = r.strip()
                if s_stripped and r_stripped:
                    src_texts.append(s_stripped)
                    ref_texts.append(r_stripped)
        total_samples = len(src_texts)
        print(f"📊 样本数: {total_samples}")
    else:
        # 直接从本地文本读取（不尝试连接HuggingFace）
        de_path = os.path.join(data_dir, f"{split}.de")
        en_path = os.path.join(data_dir, f"{split}.en")
        if not (os.path.exists(de_path) and os.path.exists(en_path)):
            print(f"❌ 无法获取参考：请确保存在 {split}.de 和 {split}.en")
            return 0.0
        print("从本地平行文本加载参考...")
        with open(de_path, 'r', encoding='utf-8') as f_de, open(en_path, 'r', encoding='utf-8') as f_en:
            for de_line, en_line in zip(f_de, f_en):
                de_txt = de_line.strip()
                en_txt = en_line.strip()
                if de_txt and en_txt:
                    src_texts.append(de_txt)
                    ref_texts.append(en_txt)
        print(f"📊 样本数: {len(src_texts)}")
    
    # 批量生成翻译
    print(f"生成翻译 ({method}) ...")
    predictions = []
    model.eval()
    
    with torch.no_grad():
        iterator = range(0, len(src_texts), batch_size)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=(len(src_texts) + batch_size - 1) // batch_size, desc="推理中")
        for i in iterator:
            batch_texts = src_texts[i:i+batch_size]
            
            # 根据方法选择合适的参数
            if method == 'beam':
                batch_preds = inference.translate_batch(
                    batch_texts, method, max_length, beam_size, length_penalty
                )
            else:
                batch_preds = inference.translate_batch(
                    batch_texts, method, max_length
                )
            
            predictions.extend(batch_preds)
            
            if not HAS_TQDM:
                # 显示进度（无tqdm时）
                progress = min(i + batch_size, len(src_texts))
                print(f"进度: {progress}/{len(src_texts)} ({100*progress/len(src_texts):.1f}%)")
    
    # 计算BLEU分数（直接使用 tokenizer.decode 的输出）
    print("计算BLEU...")
    if predictions and ref_texts:
        bleu = sacrebleu.corpus_bleu(predictions, [ref_texts], tokenize='13a')
        print(f"✅ BLEU分数: {bleu.score:.2f}")
        
        # 显示一些翻译示例
        print("\n📝 翻译示例:")
        for i in range(min(3, len(predictions))):
            print(f"源文: {src_texts[i]}")
            print(f"参考: {ref_texts[i]}")
            print(f"翻译: {predictions[i]}")
            print("-" * 50)
        
        return bleu.score
    else:
        print("❌ 翻译结果为空")
        return 0.0

def main():
    # 加载配置作为默认值
    config = get_config()
    
    parser = argparse.ArgumentParser(description='Transformer模型评估')
    parser.add_argument('--model', required=True, help='模型检查点路径')
    parser.add_argument('--data_dir', default=config['data_dir'], help='数据目录')
    parser.add_argument('--batch_size', type=int, default=config['eval_batch_size'], 
                       help=f'推理批处理大小 (默认: {config["eval_batch_size"]}, 根据GPU调整)')
    parser.add_argument('--method', choices=['greedy', 'beam'], default=config['eval_method'], 
                       help=f'解码方法 (默认: {config["eval_method"]}): greedy (快速) 或 beam (高质量)')
    parser.add_argument('--beam_size', type=int, default=config['eval_beam_size'], 
                       help=f'束搜索大小 (默认: {config["eval_beam_size"]}, 仅beam模式)')
    parser.add_argument('--max_length', type=int, default=config['eval_max_length'], 
                       help=f'最大生成长度 (默认: {config["eval_max_length"]})')
    parser.add_argument('--length_penalty', type=float, default=config['eval_length_penalty'], 
                       help=f'长度惩罚系数 (默认: {config["eval_length_penalty"]}, tensor2tensor官方设置)')
    parser.add_argument('--split', choices=['valid', 'test'], default='test',
                       help='评估数据拆分: valid(快) 或 test(最终报告)')
    
    args = parser.parse_args()
    
    print("🎯 独立评估脚本 (测试集: newstest2014)")
    print("=" * 50)
    
    # 检查文件存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    try:
        # 加载模型
        model, src_tokenizer, tgt_tokenizer, vocab_info, device = load_model_and_tokenizers(
            args.model, args.data_dir
        )
        
        # 评估BLEU（完整测试集）
        bleu_score = evaluate_bleu(
            model, src_tokenizer, tgt_tokenizer, vocab_info, device,
            args.data_dir, args.batch_size, args.method, args.beam_size, args.max_length, args.length_penalty, args.split
        )
        
        print(f"\n🏆 最终BLEU分数: {bleu_score:.2f}")
        print(f"📊 评估配置:")
        print(f"   解码方法: {args.method}")
        print(f"   测试集: 完整测试集 (newstest2014)")
        print(f"   批大小: {args.batch_size}")
        print(f"   最大长度: {args.max_length}")
        if args.method == 'beam':
            print(f"   束搜索大小: {args.beam_size}")
            print(f"   长度惩罚: {args.length_penalty} (tensor2tensor官方设置)")
            print(f"   ⭐ 使用束搜索 (更高质量但速度较慢)")
        else:
            print(f"   ⚡ 使用贪心解码 (速度快)")
        print(f"\n💡 提示: 可通过修改 config.py 更改默认评估参数")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
