"""最小而完整的 Transformer 实现（Encoder-Decoder）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    手工实现缩放点积注意力
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, query, key, value, mask=None):
        # Q: [batch, heads, seq_len, d_k]
        d_k = query.size(-1)
        
        # 计算注意力分数: QK^T / √d_k (添加数值稳定性保护)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用mask - 按tensor2tensor实现
        if mask is not None:
            # mask中False的位置表示需要mask掉；使用~mask对布尔mask取反
            # 兼容 FP16: 使用 -1e4 而不是 -1e9 (FP16 最小值约 -65504，-1e9 会溢出)
            scores = scores.masked_fill(~mask, -1e4)
        
        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - 严格按照论文实现
    
    维度变化:
    输入: (B, S, D) -> QKV线性投影: (B, S, D) -> 重塑多头: (B, S, h, d_k) 
    -> 转置: (B, h, S, d_k) -> 注意力: (B, h, S, d_k) -> 合并: (B, S, D)
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # QKV线性投影层 (按原文tensor2tensor实现，使用bias)
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)  
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        
        self.attention = ScaledDotProductAttention(dropout_rate)
        # 不在这里额外做子层内dropout，避免重复
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q_seq_len = query.size(1)
        k_seq_len = key.size(1) 
        v_seq_len = value.size(1)
        
        # 1. QKV线性投影: (B, S, D) -> (B, S, D)
        Q = self.W_q(query)  
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 重塑多头 - 使用各自的序列长度: (B, S, D) -> (B, h, S, d_k)
        Q = Q.view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2) 
        V = V.view(batch_size, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (B, h, q_len, d_k), K,V: (B, h, k_len, d_k)
        
        # 3. 多头注意力计算 - 支持不同序列长度
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 4. 合并多头 - 输出长度跟Query一致: (B, h, q_len, d_k) -> (B, q_len, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 5. 输出投影: (B, q_len, D) -> (B, q_len, D)
        output = self.W_o(attn_output)
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    手工实现位置前馈网络
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    根据原论文和tensor2tensor实现，在ReLU后应用dropout
    """
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True)
        self.W2 = nn.Linear(d_ff, d_model, bias=True)
        # 原论文在FFN内部ReLU后有dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # 第一层线性变换 + ReLU + Dropout（按原论文）
        hidden = self.dropout(F.relu(self.W1(x)))
        # 第二层线性变换
        output = self.W2(hidden)
        return output


class PositionalEncoding(nn.Module):
    """
    手工实现位置编码 - 使用sin/cos公式
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算div_term: 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # 应用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        
        pe = pe.unsqueeze(0)  # 添加batch维度
        self.register_buffer('pe', pe)  # 注册为buffer，不参与梯度计算
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerEmbedding(nn.Module):
    """
    完整的Transformer嵌入层
    Embedding + Positional Encoding + Dropout + LayerNorm
    
    注意：在 Embedding + PE + Dropout 后添加 LayerNorm 是社区广泛采用的
    稳定化技巧（参考 attention-is-all-you-need-pytorch 等流行实现）。
    这有助于稳定小 batch size 下的训练。
    """
    def __init__(self, vocab_size, d_model, max_len=5000, dropout_rate=0.1, padding_idx=None):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Token嵌入（不在这里 scale，在 output projection 时 scale）
        # 参考 attention-is-all-you-need-pytorch 的 scale_emb_or_prj='prj' 策略
        emb = self.embedding(x) * math.sqrt(self.d_model)
        
        # 强制 mask 掉 padding 的 embedding，防止梯度污染
        # 虽然 Attention 机制会 mask，但 Residual 和 FFN 会受到 Pad 向量非零值的影响
        # 这在 Tensor2Tensor 中是默认行为 (zero_pad=True)
        if self.embedding.padding_idx is not None:
             mask = (x != self.embedding.padding_idx).unsqueeze(-1)
             emb = emb * mask
        
        # 添加位置编码
        emb = self.positional_encoding(emb)
        
        # Dropout
        emb = self.dropout(emb)
        
        return emb


class EncoderLayer(nn.Module):
    """
    Transformer Encoder层
    MultiHeadAttention + Add&Norm + FFN + Add&Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        # 1. Self-Attention + 残差连接 + Layer Norm (Post-LN)
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. Feed-Forward + 残差连接 + Layer Norm (Post-LN)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder层
    Masked Self-Attention + Cross-Attention + FFN
    每个子层都有残差连接和Layer Normalization
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # 1. Masked Self-Attention (Post-LN)
        self_attn_output, _ = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # 2. Cross-Attention (Encoder-Decoder Attention) (Post-LN)
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 3. Feed-Forward (Post-LN)
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型 - 严格按照论文实现
    所有核心组件都是手工实现，包含论文的所有关键特性
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout_rate=0.1,
        pad_token_id=0,
        src_pad_token_id=None,
        tgt_pad_token_id=None,
    ):
        super().__init__()
        
        self.d_model = d_model
        # 兼容旧参数：如果未提供分别的pad id，则两者均使用 pad_token_id
        if src_pad_token_id is None:
            src_pad_token_id = pad_token_id
        if tgt_pad_token_id is None:
            tgt_pad_token_id = pad_token_id
        self.src_pad_id = src_pad_token_id
        self.tgt_pad_id = tgt_pad_token_id
        
        # 嵌入层
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout_rate, padding_idx=self.src_pad_id)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, dropout_rate, padding_idx=self.tgt_pad_id)
        
        # Encoder层栈
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder层栈
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_decoder_layers)
        ])
        
        # 最终层归一化
        # 原论文Post-LN结构中，EncoderLayer/DecoderLayer的输出已经是LayerNorm过的
        # 再次归一化是冗余的，且可能影响梯度流
        
        # 输出投影层 - 按原文tensor2tensor实现，使用bias
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=True)
        
        # 权重初始化 - 递归应用到所有子模块
        self.apply(self._init_weights)
        
        # 权重共享 - 严格按原论文3.4节：三向共享
        # "we share the same weight matrix between the two embedding layers 
        #  and the pre-softmax linear transformation"
        # 即: src_embedding = tgt_embedding = output_projection
        self.src_embedding.embedding.weight = self.tgt_embedding.embedding.weight
        self.output_projection.weight = self.tgt_embedding.embedding.weight
        
        # output_projection.bias 置零（权重共享时的最佳实践）
        self.output_projection.bias.data.zero_()
    
    def _init_weights(self, module):
        """初始化模型权重 - 遵循论文5.3节"""
        if isinstance(module, nn.Linear):
            # 论文: 使用 Xavier 均匀初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 初始化策略：
            # 虽然论文未明确指定 Embedding 的初始化方式，但 Xavier Uniform 是 Transformer 的标准实践
            # (参考 PyTorch 官方实现及 Tensor2Tensor)
            nn.init.xavier_uniform_(module.weight)
    
    def create_padding_mask(self, seq, pad_id):
        """创建padding mask，非pad为True。返回形状 (B, 1, 1, S)。"""
        mask = (seq != pad_id).unsqueeze(1).unsqueeze(1)
        return mask
    
    def create_look_ahead_mask(self, seq_len, device=None):
        """创建look-ahead mask: (S, S) -> (1, 1, S, S)"""
        # 下三角矩阵，shape可以广播到 (B, h, S, S)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def create_decoder_mask(self, tgt_seq):
        """为decoder创建组合mask"""
        seq_len = tgt_seq.size(1)
        device = tgt_seq.device
        
        # Padding mask
        padding_mask = self.create_padding_mask(tgt_seq, self.tgt_pad_id)
        
        # Look-ahead mask
        look_ahead_mask = self.create_look_ahead_mask(seq_len, device)
        
        # 组合mask
        return padding_mask & look_ahead_mask
    
    def encode(self, src_seq):
        """Encoder前向传播"""
        # 创建padding mask
        src_mask = self.create_padding_mask(src_seq, self.src_pad_id)
        
        # 源序列嵌入
        x = self.src_embedding(src_seq)
        
        # 通过所有encoder层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        # 最终层归一化 (已移除冗余)
        
        return x, src_mask
    
    def decode(self, tgt_seq, encoder_output, src_mask):
        """Decoder前向传播"""
        # 创建decoder mask
        tgt_mask = self.create_decoder_mask(tgt_seq)
        
        # 目标序列嵌入
        x = self.tgt_embedding(tgt_seq)
        
        # 通过所有decoder层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        # 最终层归一化 (已移除冗余)
        
        return x
    
    def forward(self, src_seq, tgt_seq):
        """完整的前向传播"""
        # 1. Encode
        encoder_output, src_mask = self.encode(src_seq)
        
        # 2. Decode
        decoder_output = self.decode(tgt_seq, encoder_output, src_mask)
        
        # 3. 线性投影到词汇表
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate_step(self, src_seq, tgt_seq):
        """生成单步 - 用于推理"""
        with torch.no_grad():
            encoder_output, src_mask = self.encode(src_seq)
            decoder_output = self.decode(tgt_seq, encoder_output, src_mask)
            logits = self.output_projection(decoder_output)
            return logits[:, -1, :]  # 返回最后一个位置的logits


def create_model(config):
    """工厂函数 - 根据配置创建模型"""
    # 位置编码长度：使用标准的5000（论文默认值）
    max_len = config.get('max_position_embeddings', 5000)
    
    return Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'], 
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_len=max_len,  # 标准位置编码长度
        dropout_rate=config['dropout'],
        pad_token_id=config.get('pad_token_id', 0),
        src_pad_token_id=config.get('src_pad_token_id', config.get('pad_token_id', 0)),
        tgt_pad_token_id=config.get('tgt_pad_token_id', config.get('pad_token_id', 0))
    )


# 无 __main__ 测试块，保持文件精简
