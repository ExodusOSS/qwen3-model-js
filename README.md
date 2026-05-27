# @exodus/qwen3-model-js

Pinned Qwen3 1.7B Q8_0 GGUF: artifact metadata, SHA256 verification, and streaming downloader.

## Example

```js
import { ensurePinnedModel, getPinnedModel } from '@exodus/qwen3-model-js'

const { filename } = getPinnedModel()
await ensurePinnedModel(`./models/${filename}`, {
  onProgress: ({ fraction }) => console.log(`${(fraction * 100).toFixed(1)}%`),
})
```

## Exports

- `getPinnedModel()` — returns the frozen artifact record (`repo`, `filename`, `sha256`, `sizeBytes`, `downloadUrl`, `license`).
- `verifyPinnedModelFile(path)` — streams the file and throws on size or SHA256 mismatch.
- `ensurePinnedModel(path, opts?)` — verifies if present, otherwise downloads to a temp file, verifies, and atomically renames into place. Returns `{ downloaded }`. Opts: `onProgress`, `signal`, `timeoutMs` (default 30 min, `0` disables).
- `ModelMissingError` — thrown when the file isn't on disk.
- `ModelHashMismatchError` — thrown on size or SHA256 mismatch; carries `path`, `expected`, `actual`.
