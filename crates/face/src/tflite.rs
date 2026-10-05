//! A small TensorFlow Lite interpreter: enough of the format and the op set to run MediaPipe's
//! face detector and face mesh models (float32 activations, float16 weights) on the CPU.

use anyhow::{Context, Result, bail, ensure};

use crate::gemm::sgemm;

/// Read-only view of a flatbuffer table.
#[derive(Clone, Copy)]
struct Table<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn rd_u16(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}
fn rd_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
}
fn rd_i32(b: &[u8], p: usize) -> i32 {
    rd_u32(b, p) as i32
}

impl<'a> Table<'a> {
    fn root(buf: &'a [u8]) -> Self {
        Self { buf, pos: rd_u32(buf, 0) as usize }
    }

    fn field(&self, idx: usize) -> Option<usize> {
        let vt = (self.pos as i64 - i64::from(rd_i32(self.buf, self.pos))) as usize;
        let vt_len = rd_u16(self.buf, vt) as usize;
        let at = 4 + 2 * idx;
        if at >= vt_len {
            return None;
        }
        match rd_u16(self.buf, vt + at) {
            0 => None,
            off => Some(self.pos + off as usize),
        }
    }

    fn u8(&self, idx: usize, default: u8) -> u8 {
        self.field(idx).map_or(default, |p| self.buf[p])
    }
    fn i32(&self, idx: usize, default: i32) -> i32 {
        self.field(idx).map_or(default, |p| rd_i32(self.buf, p))
    }
    fn u32(&self, idx: usize, default: u32) -> u32 {
        self.field(idx).map_or(default, |p| rd_u32(self.buf, p))
    }

    fn table(&self, idx: usize) -> Option<Table<'a>> {
        self.field(idx).map(|p| Table { buf: self.buf, pos: p + rd_u32(self.buf, p) as usize })
    }

    /// (start, len) of a vector field.
    fn vector(&self, idx: usize) -> Option<(usize, usize)> {
        self.field(idx).map(|p| {
            let v = p + rd_u32(self.buf, p) as usize;
            (v + 4, rd_u32(self.buf, v) as usize)
        })
    }

    fn i32s(&self, idx: usize) -> Vec<i32> {
        self.vector(idx).map_or_else(Vec::new, |(s, n)| (0..n).map(|i| rd_i32(self.buf, s + 4 * i)).collect())
    }

    fn bytes(&self, idx: usize) -> &'a [u8] {
        self.vector(idx).map_or(&[][..], |(s, n)| &self.buf[s..s + n])
    }

    fn tables(&self, idx: usize) -> Vec<Table<'a>> {
        self.vector(idx).map_or_else(Vec::new, |(s, n)| {
            (0..n)
                .map(|i| {
                    let p = s + 4 * i;
                    Table { buf: self.buf, pos: p + rd_u32(self.buf, p) as usize }
                })
                .collect()
        })
    }

    fn string(&self, idx: usize) -> String {
        String::from_utf8_lossy(self.bytes(idx)).into_owned()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Padding {
    Same,
    Valid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    None,
    Relu,
    Relu6,
}

#[derive(Debug)]
enum Op {
    Conv { input: usize, filter: usize, bias: Option<usize>, out: usize, stride: (usize, usize), pad: Padding, act: Act },
    Depthwise { input: usize, filter: usize, bias: Option<usize>, out: usize, stride: (usize, usize), pad: Padding, act: Act },
    Add { a: usize, b: usize, out: usize, act: Act },
    Relu { input: usize, out: usize },
    Prelu { input: usize, alpha: usize, out: usize },
    Logistic { input: usize, out: usize },
    Pad { input: usize, pads: Vec<(usize, usize)>, out: usize },
    MaxPool { input: usize, out: usize, k: (usize, usize), stride: (usize, usize), pad: Padding },
    Reshape { input: usize, out: usize },
    Concat { inputs: Vec<usize>, axis: usize, out: usize },
}

impl Op {
    fn out(&self) -> usize {
        match self {
            Op::Conv { out, .. }
            | Op::Depthwise { out, .. }
            | Op::Add { out, .. }
            | Op::Relu { out, .. }
            | Op::Prelu { out, .. }
            | Op::Logistic { out, .. }
            | Op::Pad { out, .. }
            | Op::MaxPool { out, .. }
            | Op::Reshape { out, .. }
            | Op::Concat { out, .. } => *out,
        }
    }
}

#[derive(Clone, Debug)]
struct TensorInfo {
    shape: Vec<usize>,
    name: String,
}

/// A loaded model. Weights are dequantised to f32 at load time.
pub struct Model {
    tensors: Vec<TensorInfo>,
    consts: Vec<Option<Vec<f32>>>,
    ops: Vec<Op>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

fn act(code: u8) -> Result<Act> {
    Ok(match code {
        0 => Act::None,
        1 => Act::Relu,
        3 => Act::Relu6,
        c => bail!("unsupported fused activation {c}"),
    })
}

fn padding(code: u8) -> Padding {
    if code == 1 { Padding::Valid } else { Padding::Same }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

impl Model {
    pub fn parse(buf: &[u8]) -> Result<Self> {
        ensure!(buf.len() > 8 && &buf[4..8] == b"TFL3", "not a TFLite model");
        let model = Table::root(buf);
        let codes: Vec<i32> = model
            .tables(1)
            .iter()
            .map(|c| i32::from(c.u8(0, 0) as i8).max(c.i32(3, 0)))
            .collect();
        let buffers = model.tables(4);
        let sub = *model.tables(2).first().context("model has no subgraph")?;
        let mut tensors = Vec::new();
        let mut consts: Vec<Option<Vec<f32>>> = Vec::new();
        for t in sub.tables(0) {
            let shape: Vec<usize> = t.i32s(0).iter().map(|&d| d.max(0) as usize).collect();
            let ty = t.u8(1, 0);
            let data = buffers.get(t.u32(2, 0) as usize).map(|b| b.bytes(0)).unwrap_or(&[]);
            let c = if data.is_empty() {
                None
            } else {
                Some(match ty {
                    0 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                    1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                    2 => data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect(),
                    _ => bail!("unsupported constant tensor type {ty}"),
                })
            };
            tensors.push(TensorInfo { shape, name: t.string(3) });
            consts.push(c);
        }
        let mut ops = Vec::new();
        for o in sub.tables(3) {
            let code = *codes.get(o.u32(0, 0) as usize).context("bad opcode index")?;
            let ins: Vec<i32> = o.i32s(1);
            let outs: Vec<i32> = o.i32s(2);
            let opt = o.table(4);
            let i = |k: usize| -> Result<usize> { usize::try_from(*ins.get(k).context("missing op input")?).context("optional input") };
            let opt_in = |k: usize| ins.get(k).and_then(|&x| usize::try_from(x).ok());
            let out = usize::try_from(*outs.first().context("op without output")?)?;
            let op = match code {
                6 => {
                    // DEQUANTIZE of a constant: fold it.
                    let src = i(0)?;
                    consts[out] = Some(consts[src].clone().context("dequantize of a non-constant")?);
                    continue;
                }
                3 => {
                    let o = opt.context("conv options")?;
                    Op::Conv {
                        input: i(0)?,
                        filter: i(1)?,
                        bias: opt_in(2),
                        out,
                        stride: (o.i32(2, 1) as usize, o.i32(1, 1) as usize),
                        pad: padding(o.u8(0, 0)),
                        act: act(o.u8(3, 0))?,
                    }
                }
                4 => {
                    let o = opt.context("depthwise options")?;
                    ensure!(o.i32(3, 1) == 1, "depth multiplier != 1");
                    Op::Depthwise {
                        input: i(0)?,
                        filter: i(1)?,
                        bias: opt_in(2),
                        out,
                        stride: (o.i32(2, 1) as usize, o.i32(1, 1) as usize),
                        pad: padding(o.u8(0, 0)),
                        act: act(o.u8(4, 0))?,
                    }
                }
                0 => Op::Add { a: i(0)?, b: i(1)?, out, act: act(opt.map_or(0, |o| o.u8(0, 0)))? },
                19 => Op::Relu { input: i(0)?, out },
                54 => Op::Prelu { input: i(0)?, alpha: i(1)?, out },
                14 => Op::Logistic { input: i(0)?, out },
                34 => {
                    let p = consts[i(1)?].as_ref().context("PAD needs constant paddings")?;
                    Op::Pad { input: i(0)?, pads: p.chunks_exact(2).map(|c| (c[0] as usize, c[1] as usize)).collect(), out }
                }
                17 => {
                    let o = opt.context("pool options")?;
                    Op::MaxPool {
                        input: i(0)?,
                        out,
                        k: (o.i32(4, 1) as usize, o.i32(3, 1) as usize),
                        stride: (o.i32(2, 1) as usize, o.i32(1, 1) as usize),
                        pad: padding(o.u8(0, 0)),
                    }
                }
                22 => Op::Reshape { input: i(0)?, out },
                2 => {
                    let o = opt.context("concat options")?;
                    let rank = tensors[out].shape.len() as i32;
                    let axis = o.i32(0, 0);
                    Op::Concat { inputs: ins.iter().map(|&x| x as usize).collect(), axis: (if axis < 0 { axis + rank } else { axis }) as usize, out }
                }
                c => bail!("unsupported TFLite op {c}"),
            };
            ops.push(op);
        }
        let inputs = sub.i32s(1).iter().map(|&x| x as usize).collect();
        let outputs = sub.i32s(2).iter().map(|&x| x as usize).collect();
        Ok(Self { tensors, consts, ops, inputs, outputs })
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.tensors[self.inputs[0]].shape
    }

    pub fn output_names(&self) -> Vec<String> {
        self.outputs.iter().map(|&o| self.tensors[o].name.clone()).collect()
    }

    /// Run on one NHWC input; returns the outputs in model order.
    pub fn run(&self, input: &[f32], scratch: &mut Scratch) -> Result<Vec<Vec<f32>>> {
        let n_in: usize = self.input_shape().iter().product();
        ensure!(input.len() == n_in, "input has {} values, model wants {n_in}", input.len());
        let vals = &mut scratch.vals;
        if vals.len() != self.tensors.len() {
            vals.clear();
            vals.resize(self.tensors.len(), Vec::new());
        }
        let inp = &mut vals[self.inputs[0]];
        inp.clear();
        inp.extend_from_slice(input);
        for op in &self.ops {
            self.exec(op, vals, &mut scratch.col)?;
        }
        Ok(self.outputs.iter().map(|&o| vals[o].clone()).collect())
    }

    fn get<'a>(&'a self, vals: &'a [Vec<f32>], t: usize) -> &'a [f32] {
        match &self.consts[t] {
            Some(c) => c,
            None => &vals[t],
        }
    }

    fn hwc(&self, t: usize) -> (usize, usize, usize) {
        let s = &self.tensors[t].shape;
        (s[1], s[2], s[3])
    }

    fn exec(&self, op: &Op, vals: &mut [Vec<f32>], col: &mut Vec<f32>) -> Result<()> {
        // Activation buffers are reused across ops and frames: allocating fresh multi-MB
        // vectors costs more in page faults than the arithmetic.
        let out_idx = op.out();
        let mut buf = std::mem::take(&mut vals[out_idx]);
        match op {
            Op::Conv { input, filter, bias, out, stride, pad, act } => {
                let (h, w, cin) = self.hwc(*input);
                let fs = &self.tensors[*filter].shape; // (Cout, KH, KW, Cin)
                let (cout, kh, kw) = (fs[0], fs[1], fs[2]);
                let (ho, wo, _) = self.hwc(*out);
                let (pt, pl) = (pad_before(h, ho, kh, stride.0, *pad), pad_before(w, wo, kw, stride.1, *pad));
                let x = self.get(vals, *input);
                let wt = self.get(vals, *filter);
                let k = kh * kw * cin;
                let y = &mut buf;
                fill(y, ho * wo * cout, 0.0);
                if kh == 1 && kw == 1 && stride.0 == 1 && stride.1 == 1 {
                    sgemm(ho * wo, cout, k, x, wt, y);
                } else {
                    col.clear();
                    col.resize(ho * wo * k, 0.0);
                    for oy in 0..ho {
                        for ox in 0..wo {
                            let row = &mut col[(oy * wo + ox) * k..][..k];
                            for ky in 0..kh {
                                let iy = (oy * stride.0 + ky) as isize - pt as isize;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ix = (ox * stride.1 + kx) as isize - pl as isize;
                                    if ix < 0 || ix >= w as isize {
                                        continue;
                                    }
                                    let src = &x[(iy as usize * w + ix as usize) * cin..][..cin];
                                    row[(ky * kw + kx) * cin..][..cin].copy_from_slice(src);
                                }
                            }
                        }
                    }
                    sgemm(ho * wo, cout, k, col, wt, y);
                }
                if let Some(b) = bias {
                    let b = self.get(vals, *b);
                    for px in y.chunks_exact_mut(cout) {
                        for (v, bb) in px.iter_mut().zip(b) {
                            *v += bb;
                        }
                    }
                }
                apply_act(y, *act);
            }
            Op::Depthwise { input, filter, bias, out, stride, pad, act } => {
                let (h, w, c) = self.hwc(*input);
                let fs = &self.tensors[*filter].shape; // (1, KH, KW, C)
                let (kh, kw) = (fs[1], fs[2]);
                let (ho, wo, _) = self.hwc(*out);
                let (pt, pl) = (pad_before(h, ho, kh, stride.0, *pad), pad_before(w, wo, kw, stride.1, *pad));
                let x = self.get(vals, *input);
                let wt = self.get(vals, *filter);
                let y = &mut buf;
                fill(y, ho * wo * c, 0.0);
                if let Some(b) = bias {
                    let b = self.get(vals, *b);
                    for px in y.chunks_exact_mut(c) {
                        px.copy_from_slice(b);
                    }
                }
                let row = |oy: usize, yrow: &mut [f32]| {
                    for ky in 0..kh {
                        let iy = (oy * stride.0 + ky) as isize - pt as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for ox in 0..wo {
                            let dst = &mut yrow[ox * c..][..c];
                            for kx in 0..kw {
                                let ix = (ox * stride.1 + kx) as isize - pl as isize;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let src = &x[(iy as usize * w + ix as usize) * c..][..c];
                                let wk = &wt[(ky * kw + kx) * c..][..c];
                                for ((d, s), k) in dst.iter_mut().zip(src).zip(wk) {
                                    *d += s * k;
                                }
                            }
                        }
                    }
                };
                par_rows(y, wo * c, ho * wo * c * kh * kw, row);
                apply_act(y, *act);
            }
            Op::Add { a, b, act, .. } => {
                let (x, z) = (self.get(vals, *a), self.get(vals, *b));
                ensure!(x.len() == z.len(), "ADD with broadcasting is not supported");
                let y = &mut buf;
                y.clear();
                y.extend(x.iter().zip(z).map(|(p, q)| p + q));
                apply_act(y, *act);
            }
            Op::Relu { input, .. } => {
                buf.clear();
                buf.extend(self.get(vals, *input).iter().map(|v| v.max(0.0)));
            }
            Op::Prelu { input, alpha, .. } => {
                let x = self.get(vals, *input);
                let a = self.get(vals, *alpha);
                let y = &mut buf;
                y.clear();
                y.extend_from_slice(x);
                // Branch-free so it vectorises: max(v, 0) + alpha * min(v, 0).
                for px in y.chunks_exact_mut(a.len()) {
                    for (v, &al) in px.iter_mut().zip(a) {
                        *v = v.max(0.0) + al * v.min(0.0);
                    }
                }
            }
            Op::Logistic { input, .. } => {
                buf.clear();
                buf.extend(self.get(vals, *input).iter().map(|&v| 1.0 / (1.0 + (-v).exp())));
            }
            Op::Pad { input, pads, out } => {
                let s = &self.tensors[*input].shape;
                ensure!(s.len() == 4 && pads.len() == 4 && pads[0] == (0, 0), "PAD supports NHWC only");
                let (h, w, c) = (s[1], s[2], s[3]);
                let (ho, wo, co) = self.hwc(*out);
                let x = self.get(vals, *input);
                let y = &mut buf;
                fill(y, ho * wo * co, 0.0);
                for iy in 0..h {
                    for ix in 0..w {
                        let dst = ((iy + pads[1].0) * wo + ix + pads[2].0) * co + pads[3].0;
                        y[dst..dst + c].copy_from_slice(&x[(iy * w + ix) * c..][..c]);
                    }
                }
            }
            Op::MaxPool { input, out, k, stride, pad } => {
                let (h, w, c) = self.hwc(*input);
                let (ho, wo, _) = self.hwc(*out);
                let (pt, pl) = (pad_before(h, ho, k.0, stride.0, *pad), pad_before(w, wo, k.1, stride.1, *pad));
                let x = self.get(vals, *input);
                let y = &mut buf;
                fill(y, ho * wo * c, f32::NEG_INFINITY);
                for oy in 0..ho {
                    for ox in 0..wo {
                        let dst = &mut y[(oy * wo + ox) * c..][..c];
                        for ky in 0..k.0 {
                            let iy = (oy * stride.0 + ky) as isize - pt as isize;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            for kx in 0..k.1 {
                                let ix = (ox * stride.1 + kx) as isize - pl as isize;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                for (d, s) in dst.iter_mut().zip(&x[(iy as usize * w + ix as usize) * c..][..c]) {
                                    *d = d.max(*s);
                                }
                            }
                        }
                    }
                }
            }
            Op::Reshape { input, .. } => {
                buf.clear();
                buf.extend_from_slice(self.get(vals, *input));
            }
            Op::Concat { inputs, axis, out } => {
                let os = &self.tensors[*out].shape;
                let outer: usize = os[..*axis].iter().product();
                let y = &mut buf;
                y.clear();
                for o in 0..outer {
                    for &t in inputs {
                        let inner: usize = self.tensors[t].shape[*axis..].iter().product();
                        y.extend_from_slice(&self.get(vals, t)[o * inner..][..inner]);
                    }
                }
            }
        };
        vals[out_idx] = buf;
        Ok(())
    }
}

/// Reusable buffers for `Model::run`.
#[derive(Default)]
pub struct Scratch {
    vals: Vec<Vec<f32>>,
    col: Vec<f32>,
}

/// Run `f(row_index, row)` over the rows of `y`, split across threads when the work is big.
fn par_rows(y: &mut [f32], row_len: usize, work: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(6);
    let rows = y.len() / row_len;
    if threads < 2 || work < 1 << 18 || rows < threads {
        for (i, r) in y.chunks_exact_mut(row_len).enumerate() {
            f(i, r);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|sc| {
        for (t, block) in y.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (i, r) in block.chunks_exact_mut(row_len).enumerate() {
                    f(t * per + i, r);
                }
            });
        }
    });
}

fn fill(buf: &mut Vec<f32>, n: usize, v: f32) {
    buf.clear();
    buf.resize(n, v);
}

fn pad_before(input: usize, output: usize, k: usize, stride: usize, pad: Padding) -> usize {
    match pad {
        Padding::Valid => 0,
        Padding::Same => ((output - 1) * stride + k).saturating_sub(input) / 2,
    }
}

fn apply_act(y: &mut [f32], a: Act) {
    match a {
        Act::None => {}
        Act::Relu => y.iter_mut().for_each(|v| *v = v.max(0.0)),
        Act::Relu6 => y.iter_mut().for_each(|v| *v = v.clamp(0.0, 6.0)),
    }
}

/// Files inside a MediaPipe `.task` bundle (an uncompressed zip; entries may be preceded by
/// alignment padding, so local headers are found by their signature).
pub fn read_task_entry(task: &[u8], name: &str) -> Result<Vec<u8>> {
    let find = |from: usize| (from..task.len().saturating_sub(4)).find(|&i| rd_u32(task, i) == 0x0403_4b50);
    let mut next = find(0);
    while let Some(p) = next {
        if p + 30 > task.len() {
            break;
        }
        let method = rd_u16(task, p + 8);
        let size = rd_u32(task, p + 18) as usize;
        let name_len = rd_u16(task, p + 26) as usize;
        let extra = rd_u16(task, p + 28) as usize;
        let entry = std::str::from_utf8(&task[p + 30..p + 30 + name_len]).unwrap_or("");
        let data = p + 30 + name_len + extra;
        if entry == name {
            ensure!(method == 0, "{name} is compressed; expected a stored entry");
            return Ok(task[data..data + size].to_vec());
        }
        next = find(data + size);
    }
    bail!("{name} not found in the .task bundle")
}
