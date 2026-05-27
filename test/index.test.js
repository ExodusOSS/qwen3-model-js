import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  ModelHashMismatchError,
  ModelMissingError,
  ensurePinnedModel,
  getPinnedModel,
  verifyPinnedModelFile,
} from '../index.js'

function makeTmpDir() {
  const dir = mkdtempSync(join(tmpdir(), 'qwen3-model-js-'))
  return {
    dir,
    cleanup() {
      rmSync(dir, { recursive: true, force: true })
    },
  }
}

// ─── pinned artifact ─────────────────────────────────────────────────

test('getPinnedModel returns a frozen artifact record', () => {
  const m = getPinnedModel()
  assert.equal(m.repo, 'Qwen/Qwen3-1.7B-GGUF')
  assert.equal(m.filename, 'Qwen3-1.7B-Q8_0.gguf')
  assert.equal(m.sha256.length, 64)
  assert.match(m.sha256, /^[0-9a-f]+$/)
  assert.equal(m.sizeBytes, 1834426016)
  assert.equal(m.license, 'Apache-2.0')
  assert.match(m.downloadUrl, /^https:\/\/huggingface\.co\//)
  // Mutating the artifact must fail.
  assert.throws(() => {
    m.repo = 'mutated'
  })
  assert.throws(() => {
    m.sha256 = '0'.repeat(64)
  })
})

// ─── verifyPinnedModelFile ───────────────────────────────────────────

test('verifyPinnedModelFile: ModelMissingError when file is absent', async () => {
  const t = makeTmpDir()
  try {
    await assert.rejects(verifyPinnedModelFile(join(t.dir, 'nope.gguf')), ModelMissingError)
  } finally {
    t.cleanup()
  }
})

test('verifyPinnedModelFile: ModelHashMismatchError on wrong size (cheap pre-check)', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'fake.gguf')
    writeFileSync(path, Buffer.alloc(1024))
    await assert.rejects(verifyPinnedModelFile(path), ModelHashMismatchError)
  } finally {
    t.cleanup()
  }
})

test('verifyPinnedModelFile: error carries path + expected + actual', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'fake.gguf')
    writeFileSync(path, Buffer.alloc(1024))
    try {
      await verifyPinnedModelFile(path)
      assert.fail('should have thrown')
    } catch (e) {
      assert.ok(e instanceof ModelHashMismatchError)
      assert.equal(e.path, path)
      assert.match(e.actual, /1024 bytes/)
      assert.match(e.expected, /bytes/)
    }
  } finally {
    t.cleanup()
  }
})

test('verifyPinnedModelFile: re-throws non-ENOENT fs errors', async () => {
  // Pass an unreadable path. On macOS / Linux, /dev/null/missing is
  // typically a non-ENOENT (ENOTDIR) — let the OS produce whatever
  // error it likes; we just want to confirm it's NOT a ModelMissingError.
  await assert.rejects(verifyPinnedModelFile('/dev/null/notapath'), (e) => {
    assert.ok(!(e instanceof ModelMissingError))
    return true
  })
})

// ─── ensurePinnedModel ───────────────────────────────────────────────

test('ensurePinnedModel: missing file is a hard error from the downloader (no network in tests)', async () => {
  // The download step requires real network access; we don't want
  // tests hitting HuggingFace. So we expect the call to reject — but
  // the rejection should come from the fetch path, not from the verify
  // step (which would only see the file post-download).
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'will-not-be-fetched.gguf')
    // Override fetch to throw a deterministic "no network" error so we
    // don't actually hit HF from a unit test.
    const origFetch = global.fetch
    global.fetch = async () => {
      throw new TypeError('fetch not allowed in tests')
    }

    try {
      await assert.rejects(ensurePinnedModel(path), /fetch not allowed in tests/)
    } finally {
      global.fetch = origFetch
    }
  } finally {
    t.cleanup()
  }
})

test('ensurePinnedModel: bails on hash mismatch when file already present', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'corrupt.gguf')
    writeFileSync(path, Buffer.alloc(1024))
    await assert.rejects(ensurePinnedModel(path), ModelHashMismatchError)
  } finally {
    t.cleanup()
  }
})

// ─── ensurePinnedModel (download path with mocked fetch) ─────────────
//
// downloadPinnedModel is intentionally NOT exported — the public
// download entry point is ensurePinnedModel, which always composes
// download-to-temp + verify + atomic rename. These tests exercise the
// download branches by going through that single entry point.

test('ensurePinnedModel: HTTP error becomes a thrown Error', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'wont-create.gguf')
    const origFetch = global.fetch
    global.fetch = async () => ({ ok: false, status: 503 })

    try {
      await assert.rejects(ensurePinnedModel(path), /HTTP 503/)
    } finally {
      global.fetch = origFetch
    }
  } finally {
    t.cleanup()
  }
})

test('ensurePinnedModel: missing response body throws', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'no-body.gguf')
    const origFetch = global.fetch
    global.fetch = async () => ({
      ok: true,
      status: 200,
      headers: new Map(),
      body: null,
    })

    try {
      await assert.rejects(ensurePinnedModel(path), /response had no body/)
    } finally {
      global.fetch = origFetch
    }
  } finally {
    t.cleanup()
  }
})

// Mock fetch that respects AbortSignal — matches real-fetch semantics:
// reject immediately if already aborted, otherwise hang until aborted.
function makeAbortableFetch() {
  return async (_url, opts) => {
    const signal = opts?.signal
    return new Promise((_resolve, reject) => {
      const rejectAborted = () =>
        reject(
          Object.assign(new Error(signal?.reason?.message ?? 'aborted'), {
            name: 'AbortError',
          })
        )
      if (signal?.aborted) {
        rejectAborted()
        return
      }

      signal?.addEventListener('abort', rejectAborted)
    })
  }
}

test('ensurePinnedModel: AbortSignal cancels the in-flight download', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'aborted.gguf')
    const origFetch = global.fetch
    global.fetch = makeAbortableFetch()

    const controller = new AbortController()
    // Abort before awaiting so we don't race the body of ensurePinnedModel.
    controller.abort()
    try {
      await assert.rejects(
        ensurePinnedModel(path, { signal: controller.signal, timeoutMs: 0 }),
        /aborted/
      )
    } finally {
      global.fetch = origFetch
    }
  } finally {
    t.cleanup()
  }
})

test('ensurePinnedModel: writes atomically (temp file → rename), never leaves partial at destPath', async () => {
  const t = makeTmpDir()
  try {
    // Mock fetch to return a "valid" payload that matches our fake pin.
    // We don't have the real model bytes, so we can't actually pass
    // SHA verification. Instead we verify the FAILURE path leaves no
    // garbage at destPath.
    const path = join(t.dir, 'will-be-atomic.gguf')
    const origFetch = global.fetch
    global.fetch = async () => ({
      ok: true,
      status: 200,
      headers: new Map([['content-length', '5']]),
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(Buffer.from('hello'))
          controller.close()
        },
      }),
    })

    try {
      // Hash mismatch (wrong size — 5 bytes vs 1.83 GB).
      await assert.rejects(ensurePinnedModel(path), ModelHashMismatchError)
    } finally {
      global.fetch = origFetch
    }

    // The post-failure state: destPath was never created, and the
    // temp file was cleaned up.
    const { readdirSync } = await import('node:fs')
    const remaining = readdirSync(t.dir)
    assert.equal(remaining.length, 0, `expected empty dir, got: ${remaining.join(', ')}`)
  } finally {
    t.cleanup()
  }
})

test('ensurePinnedModel: download timeout aborts cleanly with no leftover temp', async () => {
  const t = makeTmpDir()
  try {
    const path = join(t.dir, 'timeout.gguf')
    const origFetch = global.fetch
    global.fetch = makeAbortableFetch()

    try {
      await assert.rejects(ensurePinnedModel(path, { timeoutMs: 10 }), /timed out after 10ms/)
    } finally {
      global.fetch = origFetch
    }

    const { readdirSync } = await import('node:fs')
    const remaining = readdirSync(t.dir)
    assert.equal(remaining.length, 0, `expected empty dir, got: ${remaining.join(', ')}`)
  } finally {
    t.cleanup()
  }
})
