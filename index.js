// @exodus/qwen3-model-js — pinned Qwen3 1.7B Q8_0 GGUF artifact.
//
// Pure model-lifecycle concerns: artifact metadata, streaming download
// with progress, SHA256 verification, and an "ensure" helper that does
// download-if-missing + verify in one call.
//
// Node built-ins only at runtime. node-llama-cpp is declared as an
// optional peer dependency to pin a compatible runtime version without
// forcing the native install on consumers that don't need it. The
// classifier / chat-session layer is the consumer's responsibility.

import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream, mkdirSync } from 'node:fs'
import { rename, stat, unlink } from 'node:fs/promises'
import { dirname } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

const DEFAULT_DOWNLOAD_TIMEOUT_MS = 30 * 60 * 1000 // 30 minutes

// Pinned artifact. Mutating any field here without also updating sha256
// + sizeBytes is a security regression — verification would fail and the
// daemon would refuse to start (which is the intended fail-closed
// behavior on integrity mismatches).
const PINNED_MODEL = Object.freeze({
  repo: 'Qwen/Qwen3-1.7B-GGUF',
  filename: 'Qwen3-1.7B-Q8_0.gguf',
  sha256: '061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a',
  sizeBytes: 1834426016,
  downloadUrl: 'https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf',
  license: 'Apache-2.0',
})

export function getPinnedModel() {
  return PINNED_MODEL
}

// ─── errors ───────────────────────────────────────────────────────────

// Distinct from generic load failures so callers can hard-exit on hash
// mismatch (integrity problem — must not silently proceed) while
// handling everything else (missing file, network error, etc.) as they
// see fit.
export class ModelHashMismatchError extends Error {
  constructor(filepath, expected, actual) {
    super(`SHA256 mismatch for ${filepath}\n  expected: ${expected}\n  got:      ${actual}`)
    this.name = 'ModelHashMismatchError'
    this.path = filepath
    this.expected = expected
    this.actual = actual
  }
}

export class ModelMissingError extends Error {
  constructor(filepath) {
    super(`model file not found at ${filepath}`)
    this.name = 'ModelMissingError'
    this.path = filepath
  }
}

// ─── verification ─────────────────────────────────────────────────────

// Stream-hash so a 1.83 GB file doesn't get buffered into memory.
async function sha256File(filepath) {
  const hash = createHash('sha256')
  await new Promise((resolve, reject) => {
    const stream = createReadStream(filepath)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('end', resolve)
    stream.on('error', reject)
  })
  return hash.digest('hex')
}

// Hard-verify a file against the pinned record. Returns void on success;
// throws the most specific error possible so callers can act on cause:
//   - ModelMissingError      → file isn't on disk
//   - ModelHashMismatchError → wrong size or wrong hash (size pre-check
//                              short-circuits before hashing 1.83 GB)
//   - other (EACCES, EIO)    → filesystem issue, bubble up unchanged
export async function verifyPinnedModelFile(filepath) {
  let stats
  try {
    stats = await stat(filepath)
  } catch (e) {
    if (e.code === 'ENOENT') throw new ModelMissingError(filepath)
    throw e
  }

  if (stats.size !== PINNED_MODEL.sizeBytes) {
    throw new ModelHashMismatchError(
      filepath,
      `${PINNED_MODEL.sizeBytes} bytes (sha256 ${PINNED_MODEL.sha256})`,
      `${stats.size} bytes`
    )
  }

  const actual = await sha256File(filepath)
  if (actual !== PINNED_MODEL.sha256) {
    throw new ModelHashMismatchError(filepath, PINNED_MODEL.sha256, actual)
  }
}

// ─── download ─────────────────────────────────────────────────────────

// Stream the pinned URL to destPath. Internal helper, NOT exported —
// the public download path goes through ensurePinnedModel, which
// always composes download-to-temp + verify + atomic rename. Exposing
// a bare download function invited callers to skip the SHA check.
//
// Creates parent directories, writes the file with O_CREAT|O_EXCL|
// O_WRONLY (so an attacker can't pre-place a symlink at destPath to
// redirect the write, and we never silently clobber an existing file
// at that path), mode 0600.
//
// Options:
//   onProgress    ({received,total,fraction}) callback
//   signal        AbortSignal — cancels the fetch + stream
//   maxSizeBytes  hard cap on bytes written; defaults to the pinned
//                 size so a compromised CDN or content-length lie
//                 can't fill the disk before the SHA check would
//                 reject the file
async function downloadPinnedModelUnsafe(destPath, opts = {}) {
  const { onProgress, signal, maxSizeBytes = PINNED_MODEL.sizeBytes } = opts
  mkdirSync(dirname(destPath), { recursive: true })
  const res = await fetch(PINNED_MODEL.downloadUrl, { redirect: 'follow', signal })
  if (!res.ok) {
    throw new Error(`download ${PINNED_MODEL.downloadUrl}: HTTP ${res.status}`)
  }

  if (!res.body) {
    throw new Error(`download ${PINNED_MODEL.downloadUrl}: response had no body`)
  }

  const total = Number(res.headers.get('content-length') ?? PINNED_MODEL.sizeBytes)
  let received = 0
  const tap = new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength
      if (received > maxSizeBytes) {
        controller.error(
          new Error(
            `download exceeded ${maxSizeBytes} bytes (got ${received}+ bytes); aborting`
          )
        )
        return
      }

      if (onProgress) {
        onProgress({ received, total, fraction: total > 0 ? received / total : 0 })
      }

      controller.enqueue(chunk)
    },
  })
  await pipeline(
    Readable.fromWeb(res.body.pipeThrough(tap)),
    createWriteStream(destPath, { flags: 'wx', mode: 0o600 }),
    { signal }
  )
}

// ─── high-level: ensure the file is on disk and verified ──────────────

// Idempotent: if the file exists and verifies, returns immediately.
// Otherwise downloads to a sibling temp file, verifies, and atomically
// renames into destPath. This gives three guarantees:
//
//   - destPath always contains either nothing or a verified file —
//     never a partial/corrupt one (a crash mid-download leaves the
//     temp file behind, not destPath)
//   - the rename atomically replaces destPath, including the symlink
//     case (rename overwrites the symlink entry, not the file the
//     symlink points to)
//   - the temp file uses O_EXCL so an attacker can't pre-place a
//     symlink at that path either
//
// Options:
//   onProgress   forwarded to downloadPinnedModel
//   timeoutMs    hard deadline on the whole download via an internal
//                AbortController. Default 30 min (generous; a 1.83 GB
//                file at 10 Mbps takes ~25 min). Pass 0 to disable.
//   signal       caller's AbortSignal — composed with the internal
//                timeout via AbortSignal.any when both are present.
//
// On verification failure post-download the corrupt temp file is
// deleted and ModelHashMismatchError is thrown. destPath stays
// untouched (still missing).
export async function ensurePinnedModel(destPath, opts = {}) {
  try {
    await verifyPinnedModelFile(destPath)
    return { downloaded: false }
  } catch (e) {
    if (!(e instanceof ModelMissingError)) throw e
  }

  const { onProgress, signal: callerSignal, timeoutMs = DEFAULT_DOWNLOAD_TIMEOUT_MS } = opts

  // Compose timeout + caller signal. AbortSignal.any merges multiple
  // signals into one that fires on the first abort.
  const timeoutController = new AbortController()
  let timeoutHandle = null
  if (timeoutMs > 0) {
    timeoutHandle = setTimeout(
      () => timeoutController.abort(new Error(`download timed out after ${timeoutMs}ms`)),
      timeoutMs
    )
  }

  const signals = [timeoutController.signal]
  if (callerSignal) signals.push(callerSignal)
  const signal = signals.length === 1 ? signals[0] : AbortSignal.any(signals)

  // Unique temp path in the same directory, so the final rename stays
  // within the filesystem (cross-device renames aren't atomic and can
  // fail). pid + time + random keeps collisions astronomically rare;
  // combined with 'wx' in downloadPinnedModel, even a deliberate
  // collision fails closed.
  const tmpPath = `${destPath}.partial.${process.pid}.${Date.now()}.${Math.floor(Math.random() * 1e9)}`

  try {
    await downloadPinnedModelUnsafe(tmpPath, { onProgress, signal })
    await verifyPinnedModelFile(tmpPath)
    await rename(tmpPath, destPath)
    return { downloaded: true }
  } catch (e) {
    // Best-effort cleanup. The temp file may not exist (download failed
    // before any bytes hit disk) or may have been moved (rename
    // succeeded after verify but a later step threw — currently no
    // such step, but defensive). unlink() throwing ENOENT here is
    // fine; suppress.
    await unlink(tmpPath).catch(() => undefined)
    throw e
  } finally {
    if (timeoutHandle) clearTimeout(timeoutHandle)
  }
}
