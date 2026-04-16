export interface EmbeddingProviderConfig {
  provider: 'openai' | 'minimax' | 'openai-compatible';
  apiKey?: string;
  baseURL?: string;
  model: string;
  dimensions?: number;
}

const DEFAULT_OPENAI_MODEL = 'text-embedding-3-large';
const DEFAULT_OPENAI_DIMENSIONS = 1536;
const DEFAULT_MINIMAX_MODEL = 'embo-01';
const DEFAULT_MINIMAX_BASE_URL = 'https://api.minimax.io/v1';

export function loadEmbeddingProviderConfig(): EmbeddingProviderConfig | null {
  const provider = (process.env.GBRAIN_EMBEDDING_PROVIDER || '').trim().toLowerCase();

  if (provider === 'minimax' || process.env.MINIMAX_API_KEY) {
    return {
      provider: 'minimax',
      apiKey: process.env.MINIMAX_API_KEY || process.env.GBRAIN_EMBEDDING_API_KEY,
      baseURL: process.env.MINIMAX_BASE_URL || process.env.GBRAIN_EMBEDDING_BASE_URL || DEFAULT_MINIMAX_BASE_URL,
      model: process.env.MINIMAX_EMBEDDING_MODEL || process.env.GBRAIN_EMBEDDING_MODEL || DEFAULT_MINIMAX_MODEL,
      dimensions: parseOptionalInt(process.env.MINIMAX_EMBEDDING_DIMENSIONS || process.env.GBRAIN_EMBEDDING_DIMENSIONS),
    };
  }

  if (provider === 'openai-compatible') {
    return {
      provider: 'openai-compatible',
      apiKey: process.env.GBRAIN_EMBEDDING_API_KEY,
      baseURL: process.env.GBRAIN_EMBEDDING_BASE_URL,
      model: process.env.GBRAIN_EMBEDDING_MODEL || DEFAULT_OPENAI_MODEL,
      dimensions: parseOptionalInt(process.env.GBRAIN_EMBEDDING_DIMENSIONS) ?? DEFAULT_OPENAI_DIMENSIONS,
    };
  }

  if (process.env.OPENAI_API_KEY || provider === 'openai' || !provider) {
    return {
      provider: 'openai',
      apiKey: process.env.OPENAI_API_KEY || process.env.GBRAIN_EMBEDDING_API_KEY,
      baseURL: process.env.OPENAI_BASE_URL || process.env.GBRAIN_EMBEDDING_BASE_URL,
      model: process.env.OPENAI_EMBEDDING_MODEL || process.env.GBRAIN_EMBEDDING_MODEL || DEFAULT_OPENAI_MODEL,
      dimensions: parseOptionalInt(process.env.OPENAI_EMBEDDING_DIMENSIONS || process.env.GBRAIN_EMBEDDING_DIMENSIONS) ?? DEFAULT_OPENAI_DIMENSIONS,
    };
  }

  return null;
}

export function hasEmbeddingProvider(): boolean {
  const cfg = loadEmbeddingProviderConfig();
  return !!(cfg && cfg.apiKey);
}

function parseOptionalInt(value?: string): number | undefined {
  if (!value) return undefined;
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : undefined;
}
