import zstandard as zstd

class CompressionAnomaly:
    """Lens 2: negative bits-per-byte after Zstd compression
       (lower bpb → more AI-like, so we return −bpb)."""
    def __init__(self, level: int = 9):
        self.zctx = zstd.ZstdCompressor(level=level)

    def score(self, text: str) -> float:
        raw = text.encode("utf8")
        if not raw:
            return 0.0
        comp = self.zctx.compress(raw)
        bpb  = (len(comp) * 8) / len(raw)      # bits per byte
        return -bpb
