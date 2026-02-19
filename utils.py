"""
共享工具函数 - 避免代码重复
"""

import os
import torch
from tokenizers import Tokenizer
from tokenizers import decoders
try:
    import sentencepiece as spm  # type: ignore
    HAS_SPM = True
except Exception:
    HAS_SPM = False
from model import create_model


class SPWrapper:
    """SentencePiece tokenizer 包装器，提供与 HuggingFace Tokenizer 兼容的接口"""
    def __init__(self, sp):
        self.sp = sp
    
    def encode(self, text):
        """返回与 HuggingFace Tokenizer 兼容的编码结果"""
        return type('Encoding', (), {'ids': self.sp.encode(text, out_type=int)})
    
    def encode_batch(self, texts):
        """批量编码"""
        return [type('Encoding', (), {'ids': self.sp.encode(t, out_type=int)}) for t in texts]
    
    def decode(self, ids):
        """解码 token IDs 为文本"""
        return self.sp.decode(ids)
    
    def get_vocab_size(self):
        """获取词汇表大小"""
        return self.sp.get_piece_size()
    
    def token_to_id(self, token):
        """将 token 转换为 ID - 使用 SentencePiece 内置方法获取特殊 token"""
        # 特殊 token 必须使用 SentencePiece 的内置方法，而不是 piece_to_id
        # 因为 piece_to_id('<pad>') 可能返回与 pad_id() 不同的值！
        if token == '<pad>':
            return self.sp.pad_id()
        elif token == '<unk>':
            return self.sp.unk_id()
        elif token == '<s>':
            return self.sp.bos_id()
        elif token == '</s>':
            return self.sp.eos_id()
        else:
            return self.sp.piece_to_id(token)


def _ensure_tokenizer_decoder(tok) -> None:
    """Attach a reasonable decoder if missing, based on vocab markers.
    
    仅适用于HuggingFace Tokenizer。SentencePiece等有内置decode方法，无需处理。
    """
    # 检查是否是HuggingFace Tokenizer（有get_vocab方法）
    if not hasattr(tok, 'get_vocab'):
        return  # SentencePiece wrapper等，跳过
    
    try:
        if getattr(tok, 'decoder', None) is not None:
            return
    except Exception:
        pass
    try:
        vocab = tok.get_vocab()
        keys = list(vocab.keys())
    except Exception:
        keys = []
    # Heuristics: 'Ġ' for ByteLevel BPE (GPT-2/RoBERTa-like) - 优先检测
    if any('Ġ' in k for k in keys):
        tok.decoder = decoders.ByteLevel()
        return
    # '##' for WordPiece (BERT-like)
    if any(k.startswith('##') for k in keys):
        tok.decoder = decoders.WordPiece(prefix='##')
        return
    # Fallback: 默认使用ByteLevel（更通用）
        tok.decoder = decoders.ByteLevel()
    return

def load_model_and_tokenizers(checkpoint_path: str, data_dir: str, device: torch.device = None):
    """
    加载模型和分词器的统一函数
    避免在inference.py和evaluate.py中重复代码
    """
    print(f"加载检查点: {checkpoint_path}")
    
    # 设备选择
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocab_info = checkpoint['vocab_info']
    
    # 创建模型配置
    model_config = {
        'src_vocab_size': vocab_info['src_vocab_size'],
        'tgt_vocab_size': vocab_info['tgt_vocab_size'],
        'd_model': config['d_model'],
        'num_heads': config['num_heads'],
        'num_encoder_layers': config['num_encoder_layers'],
        'num_decoder_layers': config['num_decoder_layers'],
        'd_ff': config['d_ff'],
        'dropout': config['dropout'],
        'pad_token_id': vocab_info.get('pad_token_id', 0),
        'src_pad_token_id': vocab_info.get('src_pad_token_id', vocab_info.get('pad_token_id', 0)),
        'tgt_pad_token_id': vocab_info.get('tgt_pad_token_id', vocab_info.get('pad_token_id', 0))
    }
    
    # 创建并加载模型
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载分词器 - 根据checkpoint中保存的配置决定加载顺序
    shared_path = os.path.join(data_dir, 'tokenizer_shared.json')
    spm_model = os.path.join(data_dir, 'spm_shared.model')
    de_path = os.path.join(data_dir, 'tokenizer_de.json')
    en_path = os.path.join(data_dir, 'tokenizer_en.json')
    
    # 读取训练时使用的分词器后端，确保推理时使用相同的分词器
    tok_backend = config.get('tokenizer_backend', 'bpe')
    
    src_tokenizer = None
    tgt_tokenizer = None
    
    # 按照训练时的配置优先加载对应的分词器
    if tok_backend == 'sentencepiece' and HAS_SPM and os.path.exists(spm_model):
        # SentencePiece格式（训练配置优先）
        sp = spm.SentencePieceProcessor(model_file=spm_model)
        src_tokenizer = SPWrapper(sp)
        tgt_tokenizer = src_tokenizer
        print(f"✅ 加载 SentencePiece 分词器: {spm_model}")
    elif tok_backend == 'bpe' and os.path.exists(shared_path):
        # HuggingFace Tokenizers JSON格式
        src_tokenizer = Tokenizer.from_file(shared_path)
        tgt_tokenizer = src_tokenizer
        print(f"✅ 加载 HuggingFace BPE 分词器: {shared_path}")
    
    # Fallback: 如果配置的分词器不存在，尝试其他可用的
    if src_tokenizer is None:
        if HAS_SPM and os.path.exists(spm_model):
            sp = spm.SentencePieceProcessor(model_file=spm_model)
            src_tokenizer = SPWrapper(sp)
            tgt_tokenizer = src_tokenizer
            print(f"⚠️ Fallback: 加载 SentencePiece 分词器: {spm_model}")
        elif os.path.exists(shared_path):
            src_tokenizer = Tokenizer.from_file(shared_path)
            tgt_tokenizer = src_tokenizer
            print(f"⚠️ Fallback: 加载 HuggingFace BPE 分词器: {shared_path}")
        elif os.path.exists(de_path) and os.path.exists(en_path):
            src_tokenizer = Tokenizer.from_file(de_path)
            tgt_tokenizer = Tokenizer.from_file(en_path)
            print(f"⚠️ Fallback: 加载分离的德语/英语分词器")
    
    if src_tokenizer is None:
        raise FileNotFoundError(f"未找到分词器文件。请确保存在以下之一：\n"
                               f"  - {spm_model} (SentencePiece)\n"
                               f"  - {shared_path} (HuggingFace BPE)\n"
                               f"  - {de_path} 和 {en_path}")
    # 运行时确保 decoder 存在（兼容远端预置词表）
    _ensure_tokenizer_decoder(src_tokenizer)
    if tgt_tokenizer is not src_tokenizer:
        _ensure_tokenizer_decoder(tgt_tokenizer)
    
    # ---------------------------------------------------------
    # 安全检查：验证模型权重与分词器词表大小是否匹配
    # ---------------------------------------------------------
    current_src_vocab = src_tokenizer.get_vocab_size()
    current_tgt_vocab = tgt_tokenizer.get_vocab_size()
    
    model_src_vocab = vocab_info.get('src_vocab_size', config.get('src_vocab_size'))
    model_tgt_vocab = vocab_info.get('tgt_vocab_size', config.get('tgt_vocab_size'))
    
    if model_src_vocab and model_src_vocab != current_src_vocab:
        print(f"\n⚠️  警告: 词表大小不匹配！(Source)")
        print(f"   Checkpoint: {model_src_vocab}")
        print(f"   Tokenizer : {current_src_vocab}")
        print(f"   这会导致严重的翻译错误（乱码/不通顺）。建议重新训练。")
        
    if model_tgt_vocab and model_tgt_vocab != current_tgt_vocab:
        print(f"\n⚠️  警告: 词表大小不匹配！(Target)")
        print(f"   Checkpoint: {model_tgt_vocab}")
        print(f"   Tokenizer : {current_tgt_vocab}")
        print(f"   这会导致严重的翻译错误（乱码/不通顺）。建议重新训练。")
    # ---------------------------------------------------------

    print("模型与分词器就绪")
    
    return model, src_tokenizer, tgt_tokenizer, vocab_info, device
