"""
Transformer训练脚本 - 完整实现论文训练策略
手工实现所有训练相关功能，不使用高级封装

实现功能：
✅ Label Smoothing损失函数 (手工实现)
✅ Adam优化器 + 学习率调度 (Warmup策略，按论文公式)
✅ 梯度裁剪
✅ 完整的训练/验证循环
✅ 检查点保存/加载
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import sacrebleu  # 移除未使用的导入
from torch.optim.lr_scheduler import LambdaLR

# 导入AMP相关功能
# try:
#     # 优先使用新API (PyTorch 2.0+)
#     from torch.amp import autocast, GradScaler
#     HAS_AMP = True
#     USE_NEW_AMP_API = True
# except ImportError:
#     try:
#         # 回退到旧API
#         from torch.cuda.amp import autocast, GradScaler
#         HAS_AMP = True
#         USE_NEW_AMP_API = False
#     except ImportError:
#         HAS_AMP = False
#         USE_NEW_AMP_API = False

# 可选的进度条支持
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from model import create_model
from data import create_data_loaders
from config import get_config, print_config
 

"""训练器使用 LambdaLR 实现 warmup + inverse sqrt 学习率调度。"""


class Trainer:
    """Transformer训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建数据加载器
        print("🔄 加载数据...")
        self.train_loader, self.val_loader, self.vocab_info = create_data_loaders(config)
        print("✅ 数据加载器就绪")
        
        # 创建模型
        print("🔄 创建模型...")
        model_config = {
            'src_vocab_size': self.vocab_info['src_vocab_size'],
            'tgt_vocab_size': self.vocab_info['tgt_vocab_size'],
            'd_model': config['d_model'],
            'num_heads': config['num_heads'],
            'num_encoder_layers': config['num_encoder_layers'],
            'num_decoder_layers': config['num_decoder_layers'],
            'd_ff': config['d_ff'],
            'dropout': config['dropout'],
            'pad_token_id': self.vocab_info['pad_token_id'],
            'src_pad_token_id': self.vocab_info.get('src_pad_token_id', self.vocab_info['pad_token_id']),
            'tgt_pad_token_id': self.vocab_info.get('tgt_pad_token_id', self.vocab_info['pad_token_id'])
        }
        
        self.model = create_model(model_config)
        self.model = self.model.to(self.device)
        print("✅ 模型就绪")
        
        # 混合精度训练支持（需在优化前初始化，供优化函数读取）
        # self.use_amp = config.get('use_amp', False) and HAS_AMP and torch.cuda.is_available()
        # self.use_new_amp_api = USE_NEW_AMP_API
        # if self.use_amp:
        #     # GradScaler 在新API下无需传入设备参数
        #     self.scaler = GradScaler() if self.use_new_amp_api else GradScaler()
        # else:
        self.use_amp = False
        self.scaler = None
        
        # 应用性能优化
        print("🔧 应用性能设置...")
        self._apply_optimizations()
        print("✅ 性能设置完成")
        
        # 手工实现Label Smoothing损失函数
        self.label_smoothing = config['label_smoothing']
        self.pad_token_id = self.vocab_info.get('tgt_pad_token_id', self.vocab_info['pad_token_id'])
        self.vocab_size = self.vocab_info['tgt_vocab_size']

        # 应用性能优化
        print("🔧 应用性能设置...")
        self._apply_optimizations()
        print("✅ 性能设置完成")
        
        # 优化器 - 严格按论文5.3节设置: β1=0.9, β2=0.98, ε=10^-9
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1.0,  # 学习率由Warmup调度器控制，初始值会被覆盖
            betas=(0.9, 0.98),    # 论文5.3节: β1=0.9, β2=0.98  
            eps=1e-9,             # 论文5.3节: ε=10^-9
            weight_decay=0.0      # 论文中未使用权重衰减
        )

        # Label Smoothing - 严格复现版 (排除 Padding 的影响)
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, padding_idx, smoothing=0.1):
                super().__init__()
                self.padding_idx = padding_idx
                self.smoothing = smoothing
                
            def forward(self, pred, gold):
                # pred: (B*T, V), gold: (B*T)
                gold = gold.contiguous().view(-1)
                n_class = pred.size(1)
                
                # 1. 创建 One-hot 分布
                one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
                
                # 2. 应用 Label Smoothing
                # 公式: (1 - ε) * one_hot + ε / (V - 1)
                # 注意：此处暂不处理 Padding，之后统一 mask
                one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
                
                # 3. 关键修正：强制将 Padding 位置的概率置零
                # 这样模型不会被训练去预测 "这个词有 ε/(V-1) 的概率是 Padding"
                # 虽然 Loss 计算时会 mask 掉 gold=padding 的行，
                # 但对于 gold!=padding 的行，其 target distribution 中 padding 位置必须为 0
                one_hot[:, self.padding_idx] = 0.0
                
                # 4. 重新归一化 (可选，但推荐)
                # 由于置零了 padding 概率，总和略小于 1，可以重新归一化，
                # 但通常直接使用即可，因为 KL 散度主要关注相对值，且 padding 概率通常极小
                # mask = torch.ones_like(one_hot)
                # mask[:, self.padding_idx] = 0
                # one_hot = one_hot / one_hot.sum(dim=1, keepdim=True)
                
                log_prb = F.log_softmax(pred, dim=1)
                
                non_pad_mask = gold.ne(self.padding_idx)
                loss = -(one_hot * log_prb).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).mean()
                return loss
        
        self.criterion = LabelSmoothingLoss(self.pad_token_id, self.label_smoothing)
        # Validation 时不用 label smoothing（和 example 一致）
        self.val_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='mean')
        
        # 学习率调度器 (使用LambdaLR实现warmup+inverse sqrt) - 严格遵循论文公式
        d_model = float(config['d_model'])
        warmup_steps = float(config['warmup_steps'])
        lr_scale = float(config.get('lr_scale', 1.0))
        
        def _lr_lambda(step: int) -> float:
            """论文公式: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))"""
            step = max(step, 1)  # 避免step=0
            
            # 计算学习率因子
            arg1 = step ** -0.5  # step^(-0.5)
            arg2 = step * (warmup_steps ** -1.5)  # step * warmup_steps^(-1.5)
            
            return lr_scale * (d_model ** -0.5) * min(arg1, arg2)
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
        
        # 训练状态
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 创建保存目录（使用绝对路径，避免相对路径引发混淆）
        abs_save_dir = os.path.abspath(config['save_dir'])
        os.makedirs(abs_save_dir, exist_ok=True)
        self.config['save_dir'] = abs_save_dir
        print(f"保存目录: {abs_save_dir}")
        print(f"工作目录: {os.getcwd()}")
        
        # 评估在 evaluate.py 中统一处理
        try:
            self._sanity_source_conditioning()
        except Exception as e:
            print(f"⚠️ 条件化诊断失败: {e}")

    def _sanity_source_conditioning(self):
        batch = next(iter(self.val_loader))
        src_seq = batch['src_ids'].to(self.device)
        tgt_seq = batch['tgt_ids'].to(self.device)

        decoder_input = tgt_seq[:, :-1]
        target = tgt_seq[:, 1:]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(src_seq, decoder_input)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            ).item()

            perm = torch.randperm(src_seq.size(0), device=src_seq.device)
            src_shuf = src_seq.index_select(0, perm)
            logits_shuf = self.model(src_shuf, decoder_input)
            loss_shuf = self.criterion(
                logits_shuf.reshape(-1, logits_shuf.size(-1)),
                target.reshape(-1)
            ).item()

        delta = loss_shuf - loss
        print(f"条件化诊断: loss={loss:.4f}, shuffled_src_loss={loss_shuf:.4f}, delta={delta:+.4f}")
    
    def _apply_optimizations(self):
        """应用性能优化"""
        config = self.config
        
        # 1. torch.compile优化 (PyTorch 2.0+)
        # use_compile = config.get('use_compile', False)
        # print(f"compile: {use_compile}")
        
        # if use_compile and hasattr(torch, 'compile'):
        #     compile_mode = config.get('compile_mode', 'default')
        #     print(f"torch.compile: {compile_mode}")
        #     try:
        #         self.model = torch.compile(self.model, mode=compile_mode)
        #     except Exception as e:
        #         print(f"compile失败: {e}")
        
        # 2. AMP混合精度训练
        # if self.use_amp:
        #     print(f"AMP启用 ({'amp' if self.use_new_amp_api else 'cuda.amp'})")
        
        # 3. 显示参数量
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"参数量: {self.total_params:,}")
    
    def train_step(self, batch, accumulate_batches=1):
        """训练单个batch - 支持梯度累积和AMP"""
        self.model.train()
        
        # 数据移到GPU
        src_seq = batch['src_ids'].to(self.device, non_blocking=True)
        tgt_seq = batch['tgt_ids'].to(self.device, non_blocking=True)
        
        # 准备decoder输入和目标 (按论文，输入错位1个token)
        decoder_input = tgt_seq[:, :-1]  # 去掉最后一个token
        target = tgt_seq[:, 1:]          # 去掉第一个token
        
        # 前向传播 - 支持AMP
        def _forward():
            logits = self.model(src_seq, decoder_input)
            return self.criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        
        # if self.use_amp:
        #     device_type = 'cuda' if self.use_new_amp_api and self.device.type == 'cuda' else 'cuda'
        #     ctx = autocast(device_type) if self.use_new_amp_api else autocast()
        #     with ctx:
        #         loss = _forward()
        # else:
        loss = _forward()
        
        # 反向传播 - 支持AMP
        # if self.use_amp:
        #     self.scaler.scale(loss / accumulate_batches).backward()
        # else:
        (loss / accumulate_batches).backward()
        
        return loss.item()

    def optimizer_step(self):
        """执行优化器更新和梯度清零 - 支持AMP"""
        
        # DEBUG: 检查参数是否更新 (在梯度清零前检查)
        # 每100步或者是梯度异常时都打印详细信息
        if self.step % 100 == 0:
            total_norm = 0.0
            nan_count = 0
            zero_grad_count = 0
            total_params = 0
            
            for p in self.model.parameters():
                if p.grad is not None:
                    total_params += 1
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    if torch.isnan(param_norm):
                        nan_count += 1
                    if param_norm.item() == 0.0:
                        zero_grad_count += 1
                        
            total_norm = total_norm ** 0.5
            
            # 只有当梯度范数确实为0，或者有NaN时才警告
            if total_norm == 0.0 or math.isnan(total_norm) or nan_count > 0:
                print(f"⚠️ 梯度诊断 [Step {self.step}]:")
                print(f"  Total Norm: {total_norm}")
                print(f"  NaN Grads: {nan_count}/{total_params}")
                print(f"  Zero Grads: {zero_grad_count}/{total_params}")
                print(f"  Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            elif self.step % 100 == 0:
                 # 正常情况下打印一次 Norm 确认数值范围
                 pass # print(f"  [Step {self.step}] Grad Norm: {total_norm:.4f}")
        
        # if self.use_amp:
        #     # AMP模式下的梯度裁剪和优化器更新
        #     # 1. Unscale gradients
        #     self.scaler.unscale_(self.optimizer)
        #     
        #     # 2. Clip gradients (now unscaled)
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        #     
        #     # 3. Update weights (scaler will skip if infs/NaNs are found)
        #     self.scaler.step(self.optimizer)
        #     
        #     # 4. Update scaler factor
        #     self.scaler.update()
        #     
        #     # 5. Zero grads
        #     self.optimizer.zero_grad()
        # else:
        # 常规模式 - 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 调用scheduler.step()（PyTorch 1.1.0+要求在optimizer.step()之后）
        # 注意：首次调用时PyTorch可能会发出警告，这是预期行为，可以忽略
        self.scheduler.step()
        
        # 增加步数计数（用于日志记录）
        self.step += 1
        
        # 读取当前学习率用于日志
        current_lr = self.optimizer.param_groups[0]['lr']
        return current_lr
    
    def validate(self):
        """完整验证集验证 - 提供稳定准确的loss、perplexity与token准确率"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_samples = 0
        total_tokens = 0
        correct_tokens = 0
        
        print("验证中...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                src_seq = batch['src_ids'].to(self.device)
                tgt_seq = batch['tgt_ids'].to(self.device)
                
                decoder_input = tgt_seq[:, :-1]
                target = tgt_seq[:, 1:]
                
                logits = self.model(src_seq, decoder_input)
                loss = self.val_criterion(
                    logits.reshape(-1, logits.size(-1)), 
                    target.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                total_samples += src_seq.size(0)  # batch_size

                # Token-level accuracy（忽略PAD）
                pred = logits.argmax(dim=-1)  # (B, T-1)
                non_pad = target.ne(self.pad_token_id)
                if non_pad.any():
                    correct_tokens += int((pred.eq(target) & non_pad).sum().item())
                    total_tokens += int(non_pad.sum().item())
                
                # 每100个batch显示进度（可选）
                if (batch_idx + 1) % 100 == 0:
                    current_avg = total_loss / num_batches
                    current_ppl = math.exp(min(current_avg, 10))
                    current_acc = (100.0 * correct_tokens / total_tokens) if total_tokens > 0 else 0.0
                    print(f"  验证进度: {batch_idx+1:,} 批次, 当前loss: {current_avg:.4f}, ppl: {current_ppl:.2f}, acc: {current_acc:.2f}%")
        
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 10))  # 防止溢出
        token_acc = (100.0 * correct_tokens / total_tokens) if total_tokens > 0 else 0.0
        
        print(f"验证完成: {num_batches:,} 批次, {total_samples:,} 样本")
        
        return avg_loss, perplexity, token_acc
    
    def _ensure_tokenizers(self):
        """确保分词器已加载（延迟初始化）"""
        if not hasattr(self, '_src_tokenizer'):
            from data import DataProcessor
            processor = DataProcessor(self.config['data_dir'])
            src_tok, tgt_tok = processor.load_tokenizers()
            if src_tok is None:
                raise FileNotFoundError("未找到分词器文件")
            self._src_tokenizer = src_tok
            self._tgt_tokenizer = tgt_tok
            self._eval_vocab_info = {
                'bos_token_id': tgt_tok.token_to_id('<s>'),
                'eos_token_id': tgt_tok.token_to_id('</s>'),
                'pad_token_id': tgt_tok.token_to_id('<pad>')
            }
    
    def evaluate_bleu(self, split='valid'):
        """BLEU评估"""
        try:
            from evaluate import evaluate_bleu
            self._ensure_tokenizers()
            return evaluate_bleu(
                self.model, self._src_tokenizer, self._tgt_tokenizer,
                self._eval_vocab_info, self.device, self.config['data_dir'],
                batch_size=self.config.get('eval_batch_size', 32),
                method=self.config.get('eval_method', 'beam'),
                beam_size=self.config.get('eval_beam_size', 4),
                max_length=self.config.get('eval_max_length', 100),
                length_penalty=self.config.get('eval_length_penalty', 0.6),
                split=split
            )
        except Exception as e:
            print(f"BLEU评估失败: {e}")
            return 0.0
    
    def save_checkpoint(self, filepath, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'vocab_info': self.vocab_info
        }
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")
        
        if is_best:
            best_path = os.path.join(os.path.dirname(filepath), 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 兼容新旧调度器存档
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                pass
        elif 'scheduler_step' in checkpoint:
            try:
                # 回退：仅设置last_epoch用于近似恢复
                self.scheduler.last_epoch = int(checkpoint['scheduler_step'])
            except Exception:
                pass
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"检查点已加载: {filepath}")
        print(f"恢复到 epoch {self.epoch}, step {self.step}")
    
    def train(self, num_epochs, resume_from=None):
        """完整训练流程"""
        print("🚀 开始训练...")
        print(f"设备: {self.device}")
        print(f"模型参数: {self.total_params:,}")
        print(f"训练样本: {len(self.train_loader.dataset):,}")
        print(f"验证样本: {len(self.val_loader.dataset):,}")
        
        # 恢复训练
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # 训练 - 使用梯度累积
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            accumulate_batches = self.config['accumulate_grad_batches']
            
            # 时间估算变量（简化）
            
            iterator = self.train_loader
            if HAS_TQDM:
                iterator = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(iterator):
                # 计算当前累积步骤中实际的batch数量
                batches_in_current_step = (batch_idx % accumulate_batches) + 1
                is_last_batch = batch_idx == len(self.train_loader) - 1
                
                # 如果是最后一个batch且不能整除，使用实际的batch数量
                if is_last_batch and (batch_idx + 1) % accumulate_batches != 0:
                    actual_accumulate = batches_in_current_step
                else:
                    actual_accumulate = accumulate_batches
                
                # 训练步骤 (使用正确的累积数量进行梯度缩放)
                loss = self.train_step(batch, actual_accumulate)
                total_loss += loss
                num_batches += 1
                
                # 前几个batch立即显示进度（无tqdm时）
                if not HAS_TQDM and batch_idx < 5:
                    print(f"  Batch {batch_idx+1}: Loss {loss:.4f} (累积: {batches_in_current_step}/{actual_accumulate})")
                
                # 每accumulate_batches个batch执行优化器更新
                # 注意：只在完整的累积周期结束时更新，确保梯度累积正确
                if (batch_idx + 1) % accumulate_batches == 0:
                    lr = self.optimizer_step()
                    effective_step = (batch_idx + 1) // accumulate_batches  # 正确的步数计算
                    
                    if HAS_TQDM:
                        avg_loss = total_loss / max(1, num_batches)
                        iterator.set_postfix(step=effective_step, loss=f"{loss:.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    else:
                        # 简化的进度显示
                        if effective_step % 100 == 0 or effective_step <= 10:
                            avg_loss = total_loss / num_batches
                            progress = batch_idx + 1
                            total_batches = len(self.train_loader)
                            percent = 100 * progress / total_batches
                            
                            print(f"  Step {effective_step:>4} ({percent:5.1f}%) | "
                                  f"Loss: {loss:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}")
                    
                    # 注意: 验证只在epoch结束时进行（完整验证集）
                    # 这样能提供更稳定准确的loss和perplexity指标
                
                # 每1000个batch显示进度 (避免刷屏)
                elif not HAS_TQDM and batch_idx % 1000 == 0 and batch_idx > 0:
                    progress = batch_idx + 1
                    total_batches = len(self.train_loader)
                    percent = 100 * progress / total_batches
                    avg_loss = total_loss / num_batches
                    print(f"  进度: {percent:5.1f}% ({progress:>6}/{total_batches}) | Avg Loss: {avg_loss:.4f}")
            
            # 处理epoch结束时剩余的梯度（如果有）
            if (batch_idx + 1) % accumulate_batches != 0:
                # 最后一批不足accumulate_batches，但仍需要更新
                lr = self.optimizer_step()
                effective_step = (batch_idx + 1 + accumulate_batches - 1) // accumulate_batches  # 向上取整
                print(f"  最后一批梯度更新: Step {effective_step} | LR: {lr:.2e}")
            
            train_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # 完整验证集验证 - 提供稳定准确的loss和perplexity
            val_loss, perplexity, val_acc = self.validate()
            
            # 每个epoch结束时的BLEU评估
            epoch_bleu = None
            if self.config.get('eval_bleu_per_epoch', True):
                every_n = max(1, int(self.config.get('eval_bleu_every_n_epochs', 1)))
                if ((epoch + 1) % every_n) == 0:
                    print("评估BLEU...")
                    epoch_bleu = self.evaluate_bleu('valid')
            
            print(f"Epoch {epoch + 1} 完成 ({epoch_time:.1f}s)")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  困惑度: {perplexity:.2f}")
            print(f"  Token准确率: {val_acc:.2f}%")
            if epoch_bleu is not None:
                print(f"  BLEU分数: {epoch_bleu:.2f}")
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                print(f"最佳模型更新 (loss: {val_loss:.4f})")
                # 仅保存单一best文件，避免重复占用磁盘
                best_ckpt_path = os.path.join(self.config['save_dir'], 'best_model.pt')
                self.save_checkpoint(best_ckpt_path, is_best=True)
            
            # 定期保存 - 减少保存频率，提高训练效率
            if (epoch + 1) % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pt'
                )
                self.save_checkpoint(checkpoint_path, is_best)
        
        print(f"\n🎉 训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")
        
        # 最终BLEU评估（测试集）
        print("\n📊 进行最终BLEU评估 (测试集)...")
        final_bleu = self.evaluate_bleu('test')
        if final_bleu > 0:
            print(f"最终BLEU分数: {final_bleu:.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--resume', default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    print_config()
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    epochs = args.epochs if args.epochs is not None else config.get('num_epochs', 5)
    trainer.train(epochs, args.resume)


if __name__ == "__main__":
    main()
