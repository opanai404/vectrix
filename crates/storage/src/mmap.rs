// ─────────────────────────────────────────────────────────────
// VECTRIX · storage: raw mmap wrapper
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! A minimal, safe wrapper around the POSIX `mmap` family of syscalls.
//!
//! This is the only `unsafe`-touching code in the storage crate. Every
//! method bounds-checks against the mapped length before dereferencing,
//! and `Drop` unmaps automatically. `libc` is the single dependency that
//! pulls the workspace above "zero dependencies"; the syscall surface is
//! confined to this module.

use std::fs::File;
use std::io::{self};
use std::ops::{Deref, DerefMut};
use std::os::unix::io::AsRawFd;
use std::ptr::NonNull;

/// Access mode for a mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmapMode {
    /// Read-only (`PROT_READ | MAP_PRIVATE`). Safe against concurrent
    /// readers and against corrupted files being written through.
    Read,
    /// Read-write (`PROT_READ | PROT_WRITE | MAP_SHARED`), changes
    /// visible to the backing file.
    ReadWrite,
}

/// `madvise` hints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// Expect a sequential scan; warm the page cache.
    WillNeed,
    /// Expect random access; stop prefetching.
    Random,
}

/// A memory-mapped byte region. Owns its mapping and its file handle.
#[derive(Debug)]
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
    mode: MmapMode,
    _file: File,
}

// The mapping is not a `Vec`, so `Mmap` is safe to send/share as long as
// the underlying data is not mutated concurrently with readers — which is
// guaranteed by the read-only mode contract.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Maps `len` bytes of `file` starting at offset 0.
    pub fn map(file: &File, mode: MmapMode, len: usize) -> io::Result<Mmap> {
        let prot = match mode {
            MmapMode::Read => libc::PROT_READ,
            MmapMode::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        };
        let flags = match mode {
            MmapMode::Read => libc::MAP_PRIVATE,
            MmapMode::ReadWrite => libc::MAP_SHARED,
        };

        // SAFETY: `len` is caller-checked (or zero), the fd is valid, and
        // the kernel validates the range. A null result is `MAP_FAILED`.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                prot,
                flags,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mmap` returned a valid, non-null region of `len` bytes.
        let ptr = unsafe { NonNull::new_unchecked(addr as *mut u8) };
        Ok(Mmap {
            ptr,
            len,
            mode,
            _file: file.try_clone()?,
        })
    }

    /// Borrows the mapped bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes (established at map time)
        // and the mode guarantees no writer may mutate it concurrently.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Borrows the mapped bytes mutably. Only meaningful for
    /// [`MmapMode::ReadWrite`] mappings.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` guarantees exclusive access; `ptr` is valid
        // for `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn mode(&self) -> MmapMode {
        self.mode
    }

    /// Persists dirty pages to the backing file (`msync(MS_SYNC)`).
    pub fn flush(&self) -> io::Result<()> {
        // SAFETY: pointer and length match the original mapping.
        let rc = unsafe {
            libc::msync(self.ptr.as_ptr().cast::<libc::c_void>(), self.len, libc::MS_SYNC)
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Applies a `madvise` hint.
    pub fn advise(&self, advice: Advice) -> io::Result<()> {
        let flag = match advice {
            Advice::WillNeed => libc::MADV_WILLNEED,
            Advice::Random => libc::MADV_RANDOM,
        };
        // SAFETY: pointer and length match the original mapping.
        let rc = unsafe {
            libc::madvise(self.ptr.as_ptr().cast::<libc::c_void>(), self.len, flag)
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Deref for Mmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for Mmap {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: `munmap` only needs the original (ptr, len) pair, which
        // this struct still owns; the kernel ignores errors here.
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast::<libc::c_void>(), self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn map_read_back_matches_file_contents() {
        let dir = std::env::temp_dir().join("vectrix-mmap-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        std::fs::write(&path, [1u8, 2, 3, 4, 5]).unwrap();

        let file = File::open(&path).unwrap();
        let mmap = Mmap::map(&file, MmapMode::Read, 5).unwrap();
        assert_eq!(&mmap[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn readwrite_mapping_is_visible_through_deref_mut() {
        let dir = std::env::temp_dir().join("vectrix-mmap-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rw.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        f.sync_all().unwrap();

        let file = File::open(&path).unwrap();
        let mut mmap = Mmap::map(&file, MmapMode::ReadWrite, 4).unwrap();
        mmap[0] = 0x42;
        mmap.flush().unwrap();

        let again = std::fs::read(&path).unwrap();
        assert_eq!(again[0], 0x42);
    }

    #[test]
    fn advise_is_optional_and_never_errors_on_valid_mapping() {
        let dir = std::env::temp_dir().join("vectrix-mmap-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adv.bin");
        std::fs::write(&path, vec![0u8; 64]).unwrap();

        let file = File::open(&path).unwrap();
        let mmap = Mmap::map(&file, MmapMode::Read, 64).unwrap();
        mmap.advise(Advice::WillNeed).unwrap();
        mmap.advise(Advice::Random).unwrap();
    }
}
