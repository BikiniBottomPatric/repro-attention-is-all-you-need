"""
推理脚本 - 简洁的模型推理和翻译
支持贪心解码、束搜索解码、交互式翻译
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import List
import os

from utils import load_model_and_tokenizers
try:
    from torch.amp import autocast as amp_autocast  # PyTorch 2.x
except Exception:
    try:
        from torch.cuda.amp import autocast as amp_autocast  # 1.x fallback
    except Exception:
        amp_autocast = None


class TransformerInference:
    """Transformer推理器"""
    
    def __init__(
        self,
        model,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0
    ):
        self.model = model.to(device)  # 确保model在正确设备上
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        self.model.eval()
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码输入文本 - 加上EOS（与训练格式一致）"""
        tokens = self.src_tokenizer.encode(text).ids
        # 🔧 关键修复：源序列必须加上 EOS，与训练时的数据格式一致
        # 注意：SentencePiece 如果训练时未开启 add_eos，则需要手动加。
        # 这里假设分词器没有自动加 EOS。
        if tokens and tokens[-1] != self.eos_id:
             tokens = tokens + [self.eos_id]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """解码token序列为文本"""
        # 过滤特殊token
        filtered = []
        special_tokens = [self.bos_id, self.eos_id, self.pad_id]
        for t in token_ids:
            if t not in special_tokens:
                filtered.append(t)
        return self.tgt_tokenizer.decode(filtered).strip()
    
    def greedy_decode(self, text: str, max_length: int = None) -> str:
        """贪心解码 - 单条文本（调用批量版本）"""
        results = self.greedy_decode_batch([text], max_length)
        return results[0] if results else ""

    def greedy_decode_batch(self, texts: List[str], max_length: int = None) -> List[str]:
        """批量贪心解码 - 带渐进式EOS偏置"""
        if not texts:
            return []
        from config import get_config
        config = get_config()
        if max_length is None:
            max_length = config.get('eval_max_length', 100)
        # 读取抑制与惩罚配置
        # min_len = int(config.get('min_decode_length', 3))
        # no_repeat_ngram = int(config.get('no_repeat_ngram_size', 3))
        # eos_bias = float(config.get('eos_bias', 0.0))
        # repetition_penalty = float(config.get('repetition_penalty', 1.2))
        
        # 获取UNK token ID用于抑制
        unk_id = self.tgt_tokenizer.token_to_id('<unk>') if hasattr(self.tgt_tokenizer, 'token_to_id') else 1

        with torch.no_grad():
            # 批量编码源文本并padding
            encs = self.src_tokenizer.encode_batch(texts)
            # 🔧 关键修复：源序列必须加上 EOS，与训练时的数据格式一致
            src_ids = []
            for e in encs:
                ids = e.ids
                if ids and ids[-1] != self.eos_id:
                    ids = ids + [self.eos_id]
                src_ids.append(ids)
            
            max_src_len = max((len(x) for x in src_ids), default=1)
            batch_size = len(src_ids)
            
            
            src_seq = torch.full((batch_size, max_src_len), self.pad_id, dtype=torch.long, device=self.device)
            for i, ids in enumerate(src_ids):
                if ids:
                    src_seq[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

            # 预计算encoder输出
            # if amp_autocast and self.device.type == 'cuda':
            #     with amp_autocast('cuda'):
            #         encoder_output, src_mask = self.model.encode(src_seq)
            # else:
            encoder_output, src_mask = self.model.encode(src_seq)

            # 预分配目标序列
            tgt_seq = torch.full((batch_size, max_length), self.pad_id, dtype=torch.long, device=self.device)
            tgt_seq[:, 0] = self.bos_id
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            cur_len = 1

            for _ in range(1, max_length):
                active_seq = tgt_seq[:, :cur_len]
                # if amp_autocast and self.device.type == 'cuda':
                #     with amp_autocast('cuda'):
                #         dec_out = self.model.decode(active_seq, encoder_output, src_mask)
                #         logits = self.model.output_projection(dec_out)
                # else:
                dec_out = self.model.decode(active_seq, encoder_output, src_mask)
                logits = self.model.output_projection(dec_out)

                last_logits = logits[:, -1, :].clone()
                
                # 1. 重复惩罚（逐样本）
                # if repetition_penalty != 1.0 and cur_len > 1:
                #     for i in range(batch_size):
                #         if finished[i]:
                #             continue
                #         prev_tokens = tgt_seq[i, 1:cur_len].tolist()
                #         for token_id in set(prev_tokens):
                #             if last_logits[i, token_id] > 0:
                #                 last_logits[i, token_id] = last_logits[i, token_id] / repetition_penalty
                #             else:
                #                 last_logits[i, token_id] = last_logits[i, token_id] * repetition_penalty
                
                # EOS 偏置（仅当配置中 eos_bias > 0 时生效）
                # if eos_bias > 0:
                #     last_logits[:, self.eos_id] = last_logits[:, self.eos_id] + eos_bias
                
                # 最小长度限制
                # if cur_len < min_len:
                #     last_logits[:, self.eos_id] = -1e4
                
                # 3. 已完成样本仅允许EOS
                if finished.any():
                    last_logits[finished] = -1e4  # 兼容 fp16
                    last_logits[finished, self.eos_id] = 0.0
                
                # UNK 抑制已移除（和 example 一致）
                # if unk_id is not None:
                #     last_logits[:, unk_id] = -1e4
                
                # 5. n-gram重复抑制（逐样本）
                # if no_repeat_ngram > 1 and cur_len >= no_repeat_ngram - 1:
                #     window = no_repeat_ngram - 1
                #     for i in range(batch_size):
                #         if finished[i]:
                #             continue
                #         hist = tgt_seq[i, :cur_len].tolist()
                #         ng_map = {}
                #         for j in range(len(hist) - window):
                #             prefix = tuple(hist[j:j+window])
                #             nxt = hist[j+window]
                #             s = ng_map.get(prefix)
                #             if s is None:
                #                 ng_map[prefix] = {nxt}
                #             else:
                #                 s.add(nxt)
                #         cur_prefix = tuple(hist[-window:])
                #         banned = ng_map.get(cur_prefix)
                #         if banned:
                #             last_logits[i, list(banned)] = -1e4

                next_tokens = torch.argmax(last_logits, dim=-1)
                next_tokens = torch.where(finished, torch.full_like(next_tokens, self.eos_id), next_tokens)
                tgt_seq[:, cur_len] = next_tokens
                finished |= (next_tokens == self.eos_id)
                cur_len += 1
                if bool(finished.all()):
                    break

            # 解码文本
            results: List[str] = []
            for i in range(batch_size):
                results.append(self.decode_tokens(tgt_seq[i, :cur_len].tolist()))
            return results
    
    def beam_search_decode(
        self,
        text: str,
        beam_size: int = 4,
        max_length: int = None,
        alpha: float = None
    ) -> str:
        """束搜索解码 - 单条文本（调用批量版本）"""
        results = self.beam_search_decode_batch([text], beam_size, max_length, alpha)
        return results[0] if results else ""
    
    def beam_search_decode_batch(
        self,
        texts: List[str],
        beam_size: int = 4,
        max_length: int = None,
        alpha: float = None
    ) -> List[str]:
        """跨样本×多束的批量束搜索解码（向量化实现）"""
        if not texts:
            return []
        if max_length is None:
            from config import get_config
            config = get_config()
            max_length = config.get('eval_max_length', 100)
        if alpha is None:
            from config import get_config as _gc
            _cfg = _gc()
            alpha = _cfg.get('eval_length_penalty', 0.6)

        device = self.device
        eos_id = self.eos_id
        pad_id = self.pad_id
        bos_id = self.bos_id

        with torch.no_grad():
            # 1) 批量编码源文本
            encs = self.src_tokenizer.encode_batch(texts)
            # 🔧 关键修复：源序列必须加上 EOS，与训练时的数据格式一致
            src_ids = []
            for e in encs:
                ids = e.ids
                if ids and ids[-1] != eos_id:
                    ids = ids + [eos_id]
                src_ids.append(ids)
            
            B = len(src_ids)
            S_max = max((len(x) for x in src_ids), default=1)
            
            # 直接使用 max_length，不做动态限制（和 example 一致）
            effective_max_length = max_length
            
            src_seq = torch.full((B, S_max), pad_id, dtype=torch.long, device=device)
            for i, ids in enumerate(src_ids):
                if ids:
                    src_seq[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            # 2) 预计算encoder输出
            # if amp_autocast and device.type == 'cuda':
            #     with amp_autocast('cuda'):
            #         base_enc_out, base_src_mask = self.model.encode(src_seq)
            # else:
            base_enc_out, base_src_mask = self.model.encode(src_seq)

            # 3) 初始化beam容器
            beam = int(beam_size)
            # sequences: (B, beam, 1)
            sequences = torch.full((B, beam, 1), bos_id, dtype=torch.long, device=device)
            # scores: (B, beam) - 初始仅第0束有效，其余置为极小
            scores = torch.full((B, beam), -1e4, dtype=torch.float32, device=device)
            scores[:, 0] = 0.0
            finished = torch.zeros((B, beam), dtype=torch.bool, device=device)

            # 预展开encoder输出到beam维（按需在每步flatten使用）
            # base_enc_out: (B, S, D) -> (B, beam, S, D)
            # base_src_mask: (B, 1, 1, S) -> (B, beam, 1, 1, S)
            enc_out_beam = base_enc_out.unsqueeze(1).expand(B, beam, base_enc_out.size(1), base_enc_out.size(2))
            src_mask_beam = base_src_mask.unsqueeze(1).expand(B, beam, base_src_mask.size(1), base_src_mask.size(2), base_src_mask.size(3))

            # 可选：n-gram重复抑制与最小长度
            from config import get_config as _gc2
            _cf2 = _gc2()
            # no_repeat_ngram = int(_cf2.get('no_repeat_ngram_size', 3))
            # min_len = int(_cf2.get('min_decode_length', 1))
            # eos_bias = float(_cf2.get('eos_bias', 0.0))  # EOS概率提升
            # repetition_penalty = float(_cf2.get('repetition_penalty', 1.2))  # 重复惩罚
            
            # 获取UNK token ID用于抑制
            unk_id = self.tgt_tokenizer.token_to_id('<unk>') if hasattr(self.tgt_tokenizer, 'token_to_id') else 1

            cur_len = 1
            for _ in range(1, effective_max_length):
                # 4) 准备decoder输入 (B*beam, cur_len)
                dec_in = sequences.view(B * beam, cur_len)
                enc_out_flat = enc_out_beam.contiguous().view(B * beam, enc_out_beam.size(2), enc_out_beam.size(3))
                src_mask_flat = src_mask_beam.contiguous().view(B * beam, src_mask_beam.size(2), src_mask_beam.size(3), src_mask_beam.size(4))

                # 5) 前向计算最后位置logits
                # if amp_autocast and device.type == 'cuda':
                #     with amp_autocast('cuda'):
                #         dec_out = self.model.decode(dec_in, enc_out_flat, src_mask_flat)
                #         logits = self.model.output_projection(dec_out)
                # else:
                dec_out = self.model.decode(dec_in, enc_out_flat, src_mask_flat)
                logits = self.model.output_projection(dec_out)

                last_logits = logits[:, -1, :]  # (B*beam, V)
                log_probs = F.log_softmax(last_logits, dim=-1)
                
                # EOS 偏置（仅当配置中 eos_bias > 0 时生效）
                # if eos_bias > 0:
                #     log_probs[:, eos_id] = log_probs[:, eos_id] + eos_bias
                
                # 6) 已完成束仅允许产生EOS
                finished_flat = finished.view(B * beam)
                if finished_flat.any():
                    log_probs[finished_flat] = -1e4
                    log_probs[finished_flat, eos_id] = 0.0

                # 6.1) 最小长度前禁止EOS
                # if cur_len < min_len:
                #     log_probs[:, eos_id] = -1e4

                # UNK 抑制已移除（和 example 一致）
                # if unk_id is not None:
                #     log_probs[:, unk_id] = -1e4

                # 6.2) n-gram 重复抑制（逐束处理，开销小）
                # if no_repeat_ngram > 1 and cur_len + 1 >= no_repeat_ngram:
                #     V = log_probs.size(-1)
                #     # sequences: (B, beam, cur_len)
                #     seq_flat = sequences.view(B * beam, cur_len)
                #     window = no_repeat_ngram - 1
                #     for idx in range(B * beam):
                #         hist = seq_flat[idx].tolist()
                #         # 建立已出现的 n-gram 映射: prefix -> {next}
                #         ng_map = {}
                #         for j in range(len(hist) - window):
                #             prefix = tuple(hist[j:j+window])
                #             nxt = hist[j+window]
                #             s = ng_map.get(prefix)
                #             if s is None:
                #                 ng_map[prefix] = {nxt}
                #             else:
                #                 s.add(nxt)
                #         cur_prefix = tuple(hist[-window:])
                #         banned = ng_map.get(cur_prefix)
                #         if banned:
                #             log_probs[idx, list(banned)] = -1e4
                
                # 6.3) 重复惩罚：降低已生成token的概率
                # if repetition_penalty != 1.0 and cur_len > 1:
                #     seq_flat = sequences.view(B * beam, cur_len)
                #     for idx in range(B * beam):
                #         hist = seq_flat[idx].tolist()
                #         # 跳过BOS，对已出现的token施加惩罚
                #         seen_tokens = set(hist[1:])  # 跳过BOS
                #         for token_id in seen_tokens:
                #             if log_probs[idx, token_id] > 0:
                #                 log_probs[idx, token_id] = log_probs[idx, token_id] / repetition_penalty
                #             else:
                #                 log_probs[idx, token_id] = log_probs[idx, token_id] * repetition_penalty
                
                # 7) 累积分数并选择topk
                V = log_probs.size(-1)
                log_probs = log_probs.view(B, beam, V)
                cand_scores = scores.unsqueeze(-1) + log_probs  # (B, beam, V)
                cand_scores = cand_scores.view(B, beam * V)

                topk_scores, topk_indices = torch.topk(cand_scores, k=beam, dim=-1)  # (B, beam)
                prev_beam_idx = topk_indices // V  # (B, beam)
                next_tokens = (topk_indices % V).to(torch.long)  # (B, beam)

                # 8) 组装新序列
                # 从旧序列中按prev_beam_idx选取 (gather)
                prev_seq = sequences  # (B, beam, cur_len)
                gather_idx = prev_beam_idx.unsqueeze(-1).expand(B, beam, cur_len)
                gathered = torch.gather(prev_seq, 1, gather_idx)
                sequences = torch.cat([gathered, next_tokens.unsqueeze(-1)], dim=-1)  # (B, beam, cur_len+1)

                # 9) 更新scores与finished
                scores = topk_scores
                newly_finished = next_tokens.eq(eos_id)
                finished = torch.gather(finished, 1, prev_beam_idx) | newly_finished

                cur_len += 1
                # 如果全部完成，提前结束
                if bool(finished.all()):
                    break

            # 10) 按长度惩罚选择每个样本的最佳束
            # 计算每束的有效长度：找到第一个EOS的位置（没有则为cur_len）
            seqs_flat = sequences  # (B, beam, L)
            L = seqs_flat.size(-1)
            eos_mat = seqs_flat.eq(eos_id)
            # first eos position (index), default L-1 if none, length = idx+1
            # 使用一个大索引填充未出现的位置
            eos_pos = torch.where(eos_mat.any(dim=-1), eos_mat.float().argmax(dim=-1), torch.full((B, beam), L - 1, device=device, dtype=torch.long))
            lengths = (eos_pos + 1).to(torch.float32)

            lp = ((5.0 + lengths) ** alpha) / (6.0 ** alpha)
            norm_scores = scores / lp
            best_idx = norm_scores.argmax(dim=1)  # (B,)

            # 选出最佳序列
            gather_idx = best_idx.view(B, 1, 1).expand(B, 1, L)
            best_seqs = torch.gather(seqs_flat, 1, gather_idx).squeeze(1)  # (B, L)

            # 解码到文本
            results: List[str] = []
            for i in range(B):
                results.append(self.decode_tokens(best_seqs[i].tolist()))
            return results

    def translate_batch(self, texts: List[str], method: str = 'greedy', max_length: int = None, beam_size: int = 4, alpha: float = None) -> List[str]:
        """批量翻译"""
        # 设置默认最大长度
        if max_length is None:
            from config import get_config
            config = get_config()
            max_length = config.get('eval_max_length', 100)  # 合理的默认值，避免过长生成
        if alpha is None:
            from config import get_config as _gc
            _cfg = _gc()
            alpha = _cfg.get('eval_length_penalty', 0.6)
        
        if method == 'greedy':
            return self.greedy_decode_batch(texts, max_length)
        
        return self.beam_search_decode_batch(texts, beam_size, max_length, alpha)
    
    def interactive(self, default_max_length: int = None):
        """交互式翻译 - 简化版本"""
        print("🎯 交互式翻译 (输入 'quit' 退出)")
        print("命令:")
        print("  <文本> - 贪心解码翻译")
        print("  beam:<文本> - 束搜索翻译")
        if default_max_length is None:
            from config import get_config
            config = get_config()
            default_max_length = config.get('eval_max_length', 100)
        print(f"  最大长度: {default_max_length}")
        print()
        
        while True:
            try:
                user_input = input("德语 >>> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # 简化的命令解析
                if user_input.startswith('beam:'):
                    text = user_input[5:].strip()
                    result = self.beam_search_decode(text, max_length=default_max_length)
                    method = "束搜索"
                else:
                    text = user_input
                    result = self.greedy_decode(text, max_length=default_max_length)
                    method = "贪心"
                
                print(f"英语 ({method}) >>> {result}")
                print()
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"翻译错误: {e}")



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer推理')
    parser.add_argument('--checkpoint', '--model', required=True, help='模型检查点路径')
    parser.add_argument('--data-dir', default='./wmt14_data', help='数据目录')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    parser.add_argument('--text', help='要翻译的文本')
    parser.add_argument('--input-file', help='输入文件')
    parser.add_argument('--output-file', help='输出文件')
    parser.add_argument('--method', default='greedy', choices=['greedy', 'beam'], help='解码方法')
    parser.add_argument('--beam-size', type=int, default=4, help='束大小')
    parser.add_argument('--max-length', type=int, default=100, help='最大生成长度')
    parser.add_argument('--length-penalty', type=float, default=None, help='长度惩罚 (alpha)，默认读取config')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, src_tokenizer, tgt_tokenizer, vocab_info, device = load_model_and_tokenizers(
        args.checkpoint, args.data_dir, device
    )
    
    # 创建推理器
    inference = TransformerInference(
        model, src_tokenizer, tgt_tokenizer, device,
        vocab_info['bos_token_id'], vocab_info['eos_token_id'], vocab_info['pad_token_id']
    )
    
    if args.interactive:
        # 交互模式
        inference.interactive(args.max_length)
        
    elif args.text:
        # 单文本翻译
        if args.method == 'greedy':
            result = inference.greedy_decode(args.text, args.max_length)
        else:
            result = inference.beam_search_decode(
                args.text, args.beam_size, args.max_length, args.length_penalty
            )
        
        print(f"输入: {args.text}")
        print(f"输出: {result}")
        
    elif args.input_file:
        # 文件翻译
        if not os.path.exists(args.input_file):
            print(f"❌ 文件不存在: {args.input_file}")
            return
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            # 读取文件并过滤空行 - 基础循环版本
            texts = []
            for line in f:
                line_stripped = line.strip()
                if line_stripped:
                    texts.append(line_stripped)
        
        print(f"翻译 {len(texts)} 个句子...")
        
        # 准备参数 - 基础版本
        if args.method == 'beam':
            results = inference.translate_batch(
                texts, args.method, args.max_length, args.beam_size, args.length_penalty
            )
        else:
            results = inference.translate_batch(
                texts, args.method, args.max_length
            )
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result + '\n')
            print(f"✅ 结果已保存: {args.output_file}")
        else:
            for src, tgt in zip(texts, results):
                print(f"{src} -> {tgt}")
    
    else:
        print("请指定翻译模式: --interactive, --text, 或 --input-file")


if __name__ == "__main__":
    main()
