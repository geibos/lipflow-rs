//! `C[m×n] = A[m×k] · B[n×k]ᵀ`, row-major f32.

#[cfg(target_os = "macos")]
mod imp {
    use std::os::raw::c_int;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() >= m * k && b.len() >= n * k && c.len() >= m * n, "sgemm operand sizes");
        let dim = |x: usize| c_int::try_from(x).expect("matrix dimension fits in c_int");
        // SAFETY: the asserts above guarantee every buffer covers the extents cblas reads/writes
        // for row-major A (m×k, lda=k), Bᵀ from B (n×k, ldb=k) and C (m×n, ldc=n); the slices
        // do not alias because `c` is borrowed mutably.
        unsafe {
            cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, dim(m), dim(n), dim(k), 1.0, a.as_ptr(), dim(k), b.as_ptr(), dim(k), 0.0, c.as_mut_ptr(), dim(n));
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..m {
            let row = &a[i * k..][..k];
            for j in 0..n {
                c[i * n + j] = row.iter().zip(&b[j * k..][..k]).map(|(x, y)| x * y).sum();
            }
        }
    }
}

pub use imp::sgemm;
