//! Inference building blocks matching the PyTorch modules of the vendored ESPnet.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};

/// Named weights of a PyTorch state dict, already on the target device.
pub struct Weights {
    map: HashMap<String, Tensor>,
    device: Device,
    /// Built for training: layers must use the tensors as given (some are `Var`s).
    pub trainable: bool,
}

impl Weights {
    /// Read a `torch.save(state_dict)` file. Integer tensors (`num_batches_tracked`) are dropped.
    pub fn from_pth(path: &std::path::Path, device: &Device) -> Result<Self> {
        let mut w = Self { map: HashMap::new(), device: device.clone(), trainable: false };
        w.overlay_pth(path)?;
        Ok(w)
    }

    /// All float tensors of a .safetensors file, with `strip` removed from the front of names.
    pub fn from_safetensors(path: &std::path::Path, device: &Device, strip: &str) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("reading {}", path.display()))?;
        let mut map = HashMap::new();
        for (name, t) in tensors {
            if t.dtype() == DType::F32 {
                map.insert(name.strip_prefix(strip).unwrap_or(&name).to_string(), t.to_device(device)?);
            }
        }
        Ok(Self { map, device: device.clone(), trainable: false })
    }

    /// Replace/insert tensors from a .safetensors or .pth file (e.g. the personal face model).
    pub fn overlay_file(&mut self, path: &std::path::Path) -> Result<usize> {
        if path.extension().is_some_and(|e| e == "safetensors") {
            let tensors = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("reading {}", path.display()))?;
            let mut n = 0;
            for (name, t) in tensors {
                if t.dtype() == DType::F32 {
                    self.map.insert(name, t.to_device(&self.device)?);
                    n += 1;
                }
            }
            return Ok(n);
        }
        self.overlay_pth(path)
    }

    /// Replace/insert tensors from another state dict (e.g. the personal face model).
    pub fn overlay_pth(&mut self, path: &std::path::Path) -> Result<usize> {
        let tensors = candle_core::pickle::read_all(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut n = 0;
        for (name, t) in tensors {
            if t.dtype() != DType::F32 {
                continue;
            }
            self.map.insert(name, t.to_device(&self.device)?);
            n += 1;
        }
        Ok(n)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A copy of every tensor on `device`.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let map = self.map.iter().map(|(k, t)| Ok((k.clone(), t.to_device(device)?))).collect::<Result<_>>()?;
        Ok(Self { map, device: device.clone(), trainable: self.trainable })
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.map.get(name).cloned().with_context(|| format!("missing weight {name}"))
    }

    /// Replace a tensor (e.g. with a trainable `Var`'s tensor).
    pub fn insert(&mut self, name: &str, t: Tensor) {
        self.map.insert(name.to_string(), t);
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    /// A view of the weights under `prefix.`.
    pub fn pp<'a>(&'a self, prefix: &str) -> Prefixed<'a> {
        Prefixed { w: self, prefix: prefix.to_string() }
    }
}

pub struct Prefixed<'a> {
    w: &'a Weights,
    prefix: String,
}

impl Prefixed<'_> {
    fn join(&self, name: &str) -> String {
        if self.prefix.is_empty() { name.to_string() } else { format!("{}.{name}", self.prefix) }
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.w.get(&self.join(name))
    }

    pub fn pp(&self, name: &str) -> Prefixed<'_> {
        Prefixed { w: self.w, prefix: self.join(name) }
    }

    pub fn device(&self) -> &Device {
        self.w.device()
    }

    pub fn trainable(&self) -> bool {
        self.w.trainable
    }
}

/// `nn.Linear`. The weight is kept as loaded (out, in) and used transposed, so a trainable
/// weight (a candle `Var`) sees optimizer updates.
pub struct Linear {
    w: Tensor, // (out, in)
    /// Contiguous (in, out) copy for frozen CPU inference, where a transposed operand is slow.
    wt: Option<Tensor>,
    b: Option<Tensor>,
}

impl Linear {
    pub fn load(w: &Prefixed, bias: bool) -> Result<Self> {
        let l = Self::from_weight(&w.get("weight")?, if bias { Some(w.get("bias")?) } else { None })?;
        if w.trainable() { Ok(Self { wt: None, ..l }) } else { Ok(l) }
    }

    /// From an (out, in[, 1]) weight — conv1d kernels of size 1 are linears too.
    pub fn from_weight(weight: &Tensor, b: Option<Tensor>) -> Result<Self> {
        let weight = if weight.rank() == 3 { weight.squeeze(2)? } else { weight.clone() };
        let wt = if weight.device().is_cpu() { Some(weight.t()?.contiguous()?) } else { None };
        Ok(Self { w: weight, wt, b })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (lead, din) = dims.split_at(dims.len() - 1);
        let rows: usize = lead.iter().product();
        let x2 = x.reshape((rows, din[0]))?;
        let y = match &self.wt {
            Some(wt) => x2.matmul(wt)?,
            None => x2.matmul(&self.w.t()?)?,
        };
        let y = match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut out: Vec<usize> = lead.to_vec();
        out.push(self.w.dim(0)?);
        Ok(y.reshape(out)?)
    }
}

pub struct LayerNorm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn load(w: &Prefixed, eps: f32) -> Result<Self> {
        Ok(Self { w: w.get("weight")?, b: w.get("bias")?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // The fused kernel has no backward; use the composed one inside a training graph.
        if x.track_op() || self.w.track_op() {
            return Ok(candle_nn::ops::layer_norm_slow(x, &self.w, &self.b, self.eps)?);
        }
        Ok(candle_nn::ops::layer_norm(&x.contiguous()?, &self.w, &self.b, self.eps)?)
    }
}

/// Batch norm with frozen running statistics (inference, and fine-tuning where a few clips
/// would wreck the statistics). The affine weight/bias may be trainable.
pub struct BatchNorm {
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
}

impl BatchNorm {
    pub fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self { weight: w.get("weight")?, bias: w.get("bias")?, mean: w.get("running_mean")?, var: w.get("running_var")? })
    }

    /// (scale, shift) of the equivalent per-channel affine transform.
    pub fn affine(&self) -> Result<(Tensor, Tensor)> {
        let scale = self.weight.div(&(&self.var + 1e-5)?.sqrt()?)?;
        let shift = self.bias.sub(&self.mean.mul(&scale)?)?;
        Ok((scale, shift))
    }

    /// Normalise along `channel_dim` of `x`.
    pub fn forward(&self, x: &Tensor, channel_dim: usize) -> Result<Tensor> {
        let (scale, shift) = self.affine()?;
        let mut shape = vec![1usize; x.rank()];
        shape[channel_dim] = scale.dim(0)?;
        Ok(x.broadcast_mul(&scale.reshape(shape.as_slice())?)?.broadcast_add(&shift.reshape(shape.as_slice())?)?)
    }
}

pub fn swish(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// `ff(x) = w2(relu(w1(x)))` — ESPnet's PositionwiseFeedForward.
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
}

impl FeedForward {
    pub fn load(w: &Prefixed) -> Result<Self> {
        Ok(Self { w1: Linear::load(&w.pp("w_1"), true)?, w2: Linear::load(&w.pp("w_2"), true)? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2.forward(&self.w1.forward(x)?.relu()?)
    }
}

/// Pad the last two dims of an (N, C, H, W) tensor with a constant.
pub fn pad2d(x: &Tensor, p: usize, value: f32) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    let side = Tensor::full(value, (n, c, h, p), x.device())?.to_dtype(x.dtype())?;
    let x = Tensor::cat(&[&side, x, &side], 3)?;
    let top = Tensor::full(value, (n, c, p, w + 2 * p), x.device())?.to_dtype(x.dtype())?;
    Ok(Tensor::cat(&[&top, &x, &top], 2)?)
}

/// Multi-head split: (B, T, H*dk) -> (B, H, T, dk).
pub fn heads(x: &Tensor, h: usize) -> Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    Ok(x.reshape((b, t, h, d / h))?.transpose(1, 2)?.contiguous()?)
}

/// (B, H, T, dk) -> (B, T, H*dk).
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, t, dk) = x.dims4()?;
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((b, t, h * dk))?)
}

/// Softmax over the last dim: the fused kernel for inference, the composed (differentiable)
/// form when the input is part of a training graph.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    if x.track_op() {
        return Ok(candle_nn::ops::softmax(x, D::Minus1)?);
    }
    Ok(candle_nn::ops::softmax_last_dim(&x.contiguous()?)?)
}

/// softmax(q kᵀ / √dk [+ mask]) v for (B, H, Tq, dk) × (B, H, Tk, dk).
pub fn attention_masked(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let dk = q.dim(D::Minus1)?;
    let mut scores = (q.matmul(&k.t()?)? / (dk as f64).sqrt())?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    Ok(softmax_last(&scores)?.matmul(v)?)
}

pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    attention_masked(q, k, v, None)
}

pub fn check_dim(t: &Tensor, dim: usize, want: usize, what: &str) -> Result<()> {
    let got = t.dim(dim)?;
    if got != want {
        bail!("{what}: dim {dim} is {got}, expected {want}");
    }
    Ok(())
}
