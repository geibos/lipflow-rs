//! Files in the Lipflow home, in the formats the Python app used so existing data carries over:
//! `settings.json`, `history.jsonl`, and clips as numpy `.npz` (mouth crops + text).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use crate::paths;

/// settings.json as a JSON object; unknown keys are kept as they are.
#[derive(Clone, Debug, Default)]
pub struct Settings(pub Map<String, Value>);

impl Settings {
    pub fn load() -> Self {
        std::fs::read(paths::settings())
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| match v {
                Value::Object(m) => Some(Self(m)),
                _ => None,
            })
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        let p = paths::settings();
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        // Write-then-rename so a crash never leaves half a settings file.
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&Value::Object(self.0.clone()))?)?;
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    pub fn flag(&self, key: &str, default: bool) -> bool {
        self.0.get(key).and_then(Value::as_bool).unwrap_or(default)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(Value::as_str)
    }

    pub fn set(&mut self, key: &str, v: Value) {
        self.0.insert(key.to_string(), v);
    }
}

pub fn timestamp_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis())
}

/// Local time as "%Y-%m-%dT%H:%M:%S".
pub fn local_time() -> String {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as libc_time::TimeT;
    libc_time::format_local(t)
}

mod libc_time {
    pub type TimeT = i64;

    #[repr(C)]
    struct Tm {
        tm_sec: i32,
        tm_min: i32,
        tm_hour: i32,
        tm_mday: i32,
        tm_mon: i32,
        tm_year: i32,
        tm_wday: i32,
        tm_yday: i32,
        tm_isdst: i32,
        tm_gmtoff: i64,
        tm_zone: *const std::ffi::c_char,
    }

    unsafe extern "C" {
        fn localtime_r(t: *const TimeT, out: *mut Tm) -> *mut Tm;
    }

    pub fn format_local(t: TimeT) -> String {
        let mut tm = std::mem::MaybeUninit::<Tm>::uninit();
        // SAFETY: both pointers are valid for the call; localtime_r (thread-safe) fully
        // initialises `tm` when it returns non-null, which is checked before reading it.
        let tm = unsafe {
            if localtime_r(&t, tm.as_mut_ptr()).is_null() {
                return String::new();
            }
            tm.assume_init()
        };
        format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

pub fn log_history(seconds: f64, raw: &[String], text: &str, latency: f64, cleanup: &str) -> Result<()> {
    let p = paths::history();
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let line = json!({
        "at": local_time(),
        "seconds": (seconds * 100.0).round() / 100.0,
        "raw": raw,
        "text": text,
        "latency": (latency * 100.0).round() / 100.0,
        "cleanup": cleanup,
    });
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
    writeln!(f, "{line}")?;
    Ok(())
}

// -- npz ------------------------------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut crc = !0u32;
    for &b in data {
        crc = table[((crc ^ u32::from(b)) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// A numpy array to store: header descr + shape + raw bytes.
pub struct NpyArray {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl NpyArray {
    pub fn u8(shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self { descr: "|u1".into(), shape, data }
    }

    /// numpy unicode (UTF-32LE, fixed width) — what np.savez makes of Python strings.
    pub fn strings(items: &[&str], scalar: bool) -> Self {
        let width = items.iter().map(|s| s.chars().count()).max().unwrap_or(0).max(1);
        let mut data = Vec::with_capacity(items.len() * width * 4);
        for s in items {
            let mut n = 0;
            for c in s.chars() {
                data.extend_from_slice(&(c as u32).to_le_bytes());
                n += 1;
            }
            data.extend(std::iter::repeat_n(0u8, (width - n) * 4));
        }
        Self { descr: format!("<U{width}"), shape: if scalar { Vec::new() } else { vec![items.len()] }, data }
    }

    fn to_npy(&self) -> Vec<u8> {
        let shape = match self.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.shape[0]),
            _ => format!("({})", self.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}", self.descr);
        let total = 10 + header.len() + 1;
        header.push_str(&" ".repeat((64 - total % 64) % 64));
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&self.data);
        out
    }
}

/// Write an (uncompressed) .npz: numpy reads it like savez_compressed's output.
pub fn write_npz(path: &Path, arrays: &[(&str, NpyArray)]) -> Result<()> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, arr) in arrays {
        let data = arr.to_npy();
        let fname = format!("{name}.npy");
        let crc = crc32(&data);
        let offset = out.len() as u32;
        let size = u32::try_from(data.len()).context("array too large for a zip entry")?;
        // local file header
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]); // version, flags, method=stored, time, date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(fname.as_bytes());
        out.extend_from_slice(&data);
        // central directory entry
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&size.to_le_bytes());
        central.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 12]); // extra, comment, disk, int attr, ext attr (4)
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(fname.as_bytes());
    }
    let cd_offset = out.len() as u32;
    let cd_len = central.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&(arrays.len() as u16).to_le_bytes());
    out.extend_from_slice(&(arrays.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_len.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("npz.tmp");
    std::fs::write(&tmp, &out)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Save a clip: rois (T,96,96), its text, and the model's raw guesses.
/// `frames` are either grayscale mouth crops (T×96×96, saved as "rois", English) or colour
/// face crops (T×96×96×3, saved as "faces", Russian/MultiVSR).
pub fn save_clip(dir: &Path, frames: &[u8], t: usize, text: &str, raw: &[String], extra: &[(&str, &str)]) -> Result<PathBuf> {
    let (key, shape) = if frames.len() == t * 96 * 96 {
        ("rois", vec![t, 96, 96])
    } else if frames.len() == t * 96 * 96 * 3 {
        ("faces", vec![t, 96, 96, 3])
    } else {
        bail!("clip has {} bytes, expected {t}x96x96 or {t}x96x96x3", frames.len());
    };
    let path = dir.join(format!("{}.npz", timestamp_ms()));
    let raw_refs: Vec<&str> = raw.iter().map(String::as_str).collect();
    let mut arrays = vec![(key, NpyArray::u8(shape, frames.to_vec())), ("text", NpyArray::strings(&[text], true)), ("raw", NpyArray::strings(&raw_refs, false))];
    for (k, v) in extra {
        arrays.push((k, NpyArray::strings(&[v], true)));
    }
    write_npz(&path, &arrays)?;
    Ok(path)
}

/// Keep only the newest `keep` .npz files in `dir`.
pub fn prune(dir: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut names: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "npz")).collect();
    names.sort();
    let n = names.len();
    for p in names.into_iter().take(n.saturating_sub(keep)) {
        let _ = std::fs::remove_file(p);
    }
}

/// Arrays of an .npz (stored or deflate-compressed entries), by name without ".npy".
pub fn read_npz(path: &Path) -> Result<Vec<(String, NpyArray)>> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let u16_at = |p: usize| u16::from_le_bytes([b[p], b[p + 1]]) as usize;
    let u32_at = |p: usize| u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as usize;
    let eocd = (0..b.len().saturating_sub(21)).rev().find(|&i| u32_at(i) == 0x0605_4b50).context("not a zip file")?;
    let (count, mut p) = (u16_at(eocd + 10), u32_at(eocd + 16));
    let mut out = Vec::new();
    for _ in 0..count {
        if u32_at(p) != 0x0201_4b50 {
            bail!("bad zip central directory");
        }
        let (method, csize, usize_) = (u16_at(p + 10), u32_at(p + 20), u32_at(p + 24));
        let (nlen, xlen, clen) = (u16_at(p + 28), u16_at(p + 30), u16_at(p + 32));
        let local = u32_at(p + 42);
        let name = String::from_utf8_lossy(&b[p + 46..p + 46 + nlen]).to_string();
        p += 46 + nlen + xlen + clen;
        let data_at = local + 30 + u16_at(local + 26) + u16_at(local + 28);
        let raw = b.get(data_at..data_at + csize).context("truncated zip entry")?;
        let data = match method {
            0 => raw.to_vec(),
            8 => {
                let mut v = Vec::with_capacity(usize_);
                std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(raw), &mut v)?;
                v
            }
            m => bail!("unsupported zip compression {m}"),
        };
        out.push((name.trim_end_matches(".npy").to_string(), parse_npy(&data)?));
    }
    Ok(out)
}

fn parse_npy(b: &[u8]) -> Result<NpyArray> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("not an .npy array");
    }
    let (hlen, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let header = std::str::from_utf8(&b[off..off + hlen])?;
    let descr = header.split("'descr': '").nth(1).and_then(|s| s.split('\'').next()).context("npy descr")?.to_string();
    let shape_s = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).context("npy shape")?;
    let shape = shape_s.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<Result<Vec<usize>, _>>()?;
    Ok(NpyArray { descr, shape, data: b[off + hlen..].to_vec() })
}

impl NpyArray {
    /// First string of a numpy unicode array ('<U..').
    pub fn first_string(&self) -> Option<String> {
        let width: usize = self.descr.strip_prefix("<U")?.parse().ok()?;
        let chars = self.data.get(..width * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        Some(chars.take_while(|&c| c != 0).filter_map(char::from_u32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_python_savez_compressed() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/py_clip.npz");
        let a = read_npz(&p).expect("fixture");
        let rois = &a.iter().find(|x| x.0 == "rois").expect("rois").1;
        assert_eq!(rois.shape, vec![2, 96, 96]);
        assert_eq!(rois.data[..4], [0, 1, 2, 3]);
        assert_eq!(a.iter().find(|x| x.0 == "text").and_then(|x| x.1.first_string()).as_deref(), Some("The birch canoe — slid ✓"));
    }

    #[test]
    fn writes_what_it_reads() {
        let dir = std::env::temp_dir().join(format!("lipflow-npz-{}", timestamp_ms()));
        let rois: Vec<u8> = (0..96 * 96).map(|i| (i % 251) as u8).collect();
        let p = save_clip(&dir, &rois, 1, "héllo wörld", &["RAW ONE".into(), "TWO".into()], &[("pasted", "x")]).expect("save");
        let a = read_npz(&p).expect("read back");
        assert_eq!(a.iter().find(|x| x.0 == "rois").map(|x| x.1.data.clone()), Some(rois));
        assert_eq!(a.iter().find(|x| x.0 == "text").and_then(|x| x.1.first_string()).as_deref(), Some("héllo wörld"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
