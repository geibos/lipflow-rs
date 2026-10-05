//! The small on-device cleanup model (`cleanup.py`'s "local" backend): Qwen3-0.6B, quantised
//! GGUF on candle, greedy decoding with the Qwen3 chat template (thinking off).

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use tokenizers::Tokenizer;

pub const MODEL_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";
pub const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";
pub const TOKENIZER_URL: &str = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json";

pub struct LocalLlm {
    model: ModelWeights,
    tok: Tokenizer,
    device: Device,
    stop: Vec<u32>,
}

/// Qwen3's chat template for `messages` with a generation prompt and thinking disabled.
pub fn chat_prompt(messages: &[(String, String)]) -> String {
    let mut s = String::new();
    for (role, content) in messages {
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    s
}

impl LocalLlm {
    /// `dir` holds the GGUF weights and tokenizer.json.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let path = dir.join(MODEL_FILE);
        let mut f = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let content = gguf_file::Content::read(&mut f).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
        let model = ModelWeights::from_gguf(content, &mut f, device)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();
        Ok(Self { model, tok, device: device.clone(), stop })
    }

    /// Greedy completion of a chat, at most `max_tokens` new tokens.
    pub fn chat(&mut self, messages: &[(String, String)], max_tokens: usize) -> Result<String> {
        let prompt = chat_prompt(messages);
        let enc = self.tok.encode(prompt, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        let ids = enc.get_ids().to_vec();
        self.model.clear_kv_cache();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0)?;
        let mut out: Vec<u32> = Vec::new();
        let mut pos = ids.len();
        for _ in 0..max_tokens {
            let next = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
            if self.stop.contains(&next) {
                break;
            }
            out.push(next);
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, pos)?;
            pos += 1;
        }
        self.tok.decode(&out, true).map_err(|e| anyhow!("detokenize: {e}"))
    }
}
