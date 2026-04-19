/**
 * Embedding Service
 * Ported from production Ruby implementation (embedding_service.rb, 190 LOC)
 *
 * Default provider is OpenAI text-embedding-3-large at 1536 dimensions.
 * Supports provider/base-url/model overrides for Minimax and OpenAI-compatible APIs.
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import OpenAI from 'openai';
import type { EmbeddingProviderConfig } from './provider-config.ts';
import { hasEmbeddingProvider, loadEmbeddingProviderConfig } from './provider-config.ts';

const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const DEFAULT_BATCH_SIZE = 100;
const DEFAULT_MINIMAX_REQUEST_INTERVAL_MS = 6500;

type EmbeddingKind = 'document' | 'query';
type MinimaxVector = number[] | { embedding?: number[]; vector?: number[]; values?: number[]; index?: number };
type MinimaxEmbeddingResponse = {
  vectors?: MinimaxVector[] | null;
  base_resp?: {
    status_code?: number;
    status_msg?: string;
  };
};

let client: OpenAI | null = null;
let minimaxNextRequestAt = 0;

class EmbeddingRateLimitError extends Error {
  retryAfterMs: number;

  constructor(message: string, retryAfterMs: number) {
    super(message);
    this.name = 'EmbeddingRateLimitError';
    this.retryAfterMs = retryAfterMs;
  }
}

function getClient(): OpenAI {
  if (!client) {
    const cfg = loadEmbeddingProviderConfig();
    if (!cfg?.apiKey) {
      throw new Error('No embedding provider configured. Set OPENAI_API_KEY or MINIMAX_API_KEY.');
    }
    client = new OpenAI({
      apiKey: cfg.apiKey,
      ...(cfg.baseURL ? { baseURL: cfg.baseURL } : {}),
    });
  }
  return client;
}

export async function embed(text: string): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated], 'query');
  return result[0];
}

export async function embedBatch(texts: string[], kind: EmbeddingKind = 'document'): Promise<Float32Array[]> {
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];
  const batchSize = getBatchSize();

  for (let i = 0; i < truncated.length; i += batchSize) {
    const batch = truncated.slice(i, i + batchSize);
    const batchResults = await embedBatchWithRetry(batch, kind);
    results.push(...batchResults);
  }

  return results;
}

async function embedBatchWithRetry(texts: string[], kind: EmbeddingKind): Promise<Float32Array[]> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const cfg = loadEmbeddingProviderConfig();
      if (!cfg?.apiKey) {
        throw new Error('No embedding provider configured. Set OPENAI_API_KEY or MINIMAX_API_KEY.');
      }

      if (cfg.provider === 'minimax') {
        return await embedWithMinimax(cfg, texts, kind);
      }

      const response = await getClient().embeddings.create({
        model: cfg.model,
        input: texts,
        ...(cfg.dimensions ? { dimensions: cfg.dimensions } : {}),
      });

      const sorted = response.data.sort((a, b) => a.index - b.index);
      return sorted.map(d => new Float32Array(d.embedding));
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      let delay = exponentialDelay(attempt);

      if (e instanceof EmbeddingRateLimitError) {
        delay = Math.max(delay, e.retryAfterMs);
      }

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  throw new Error('Embedding failed after all retries');
}

async function embedWithMinimax(
  cfg: EmbeddingProviderConfig,
  texts: string[],
  kind: EmbeddingKind,
): Promise<Float32Array[]> {
  await waitForMinimaxRequestSlot();

  const baseURL = (cfg.baseURL || 'https://api.minimax.io/v1').replace(/\/$/, '');
  const response = await fetch(`${baseURL}/embeddings`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${cfg.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: cfg.model,
      texts,
      type: kind === 'query' ? 'query' : 'db',
      ...(cfg.dimensions ? { dimensions: cfg.dimensions } : {}),
    }),
  });

  const payload = await response.json() as MinimaxEmbeddingResponse;
  const statusCode = payload.base_resp?.status_code ?? (response.ok ? 0 : response.status);
  if (!response.ok || statusCode !== 0) {
    const message = payload.base_resp?.status_msg || response.statusText || 'unknown error';
    if (statusCode === 1002 || /rate limit/i.test(message)) {
      throw new EmbeddingRateLimitError(
        `MiniMax embeddings failed: ${message}`,
        getMinimaxRetryAfterMs(response.headers),
      );
    }
    throw new Error(`MiniMax embeddings failed: ${message}`);
  }

  const vectors = normalizeMinimaxVectors(payload.vectors, texts.length);
  return vectors.map(v => new Float32Array(v));
}

function normalizeMinimaxVectors(vectors: MinimaxEmbeddingResponse['vectors'], expected: number): number[][] {
  if (!Array.isArray(vectors)) {
    throw new Error('MiniMax embeddings response did not include vectors');
  }

  const normalized = vectors.map((item, position) => {
    if (Array.isArray(item)) {
      return { index: position, embedding: item };
    }

    const embedding = item.embedding || item.vector || item.values;
    if (!Array.isArray(embedding)) {
      throw new Error('MiniMax embeddings response contained a vector in an unknown format');
    }

    return {
      index: typeof item.index === 'number' ? item.index : position,
      embedding,
    };
  }).sort((a, b) => a.index - b.index);

  if (normalized.length !== expected) {
    throw new Error(`MiniMax embeddings response count mismatch: expected ${expected}, got ${normalized.length}`);
  }

  return normalized.map(item => item.embedding);
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function getBatchSize(): number {
  return parsePositiveInt(process.env.GBRAIN_EMBED_BATCH_SIZE) || DEFAULT_BATCH_SIZE;
}

async function waitForMinimaxRequestSlot(): Promise<void> {
  const intervalMs = parsePositiveInt(process.env.MINIMAX_MIN_REQUEST_INTERVAL_MS)
    || parsePositiveInt(process.env.GBRAIN_MINIMAX_MIN_REQUEST_INTERVAL_MS)
    || DEFAULT_MINIMAX_REQUEST_INTERVAL_MS;
  const now = Date.now();
  const scheduledAt = Math.max(now, minimaxNextRequestAt);
  minimaxNextRequestAt = scheduledAt + intervalMs;
  const waitMs = scheduledAt - now;
  if (waitMs > 0) {
    await sleep(waitMs);
  }
}

function getMinimaxRetryAfterMs(headers: Headers): number {
  const retryAfterHeader = headers.get('retry-after');
  const parsedSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : NaN;
  if (Number.isFinite(parsedSeconds) && parsedSeconds > 0) {
    return parsedSeconds * 1000;
  }
  return 60000;
}

function parsePositiveInt(value?: string): number | undefined {
  if (!value) return undefined;
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export const EMBEDDING_MODEL = () => loadEmbeddingProviderConfig()?.model || 'text-embedding-3-large';
export const EMBEDDING_DIMENSIONS = () => loadEmbeddingProviderConfig()?.dimensions || 1536;
export { hasEmbeddingProvider };
