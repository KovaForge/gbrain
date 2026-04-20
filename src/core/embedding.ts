/**
 * Embedding Service
 * Ported from production Ruby implementation (embedding_service.rb, 190 LOC)
 *
 * Default provider is OpenAI text-embedding-3-large at 1536 dimensions.
 * Supports provider/base-url/model overrides for MiniMax and OpenAI-compatible APIs.
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import OpenAI from 'openai';
import { loadConfig, type EmbeddingProvider } from './config.ts';
import type { EmbeddingProviderConfig } from './provider-config.ts';
import { hasEmbeddingProvider, loadEmbeddingProviderConfig } from './provider-config.ts';

const OPENAI_MODEL = 'text-embedding-3-large';
const MINIMAX_MODEL = 'embo-01';
const DEFAULT_EMBEDDING_DIMENSIONS = 1536;
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
  total_tokens?: number;
  base_resp?: {
    status_code?: number;
    status_msg?: string;
  };
};

let client: OpenAI | null = null;
let openaiClient: OpenAI | null = null;
let openaiClientApiKey: string | undefined;
let minimaxNextRequestAt = 0;

interface EmbeddingConfig {
  provider: EmbeddingProvider;
  model: string;
  dimensions: number;
  apiKey?: string;
  groupId?: string;
  baseUrl?: string;
}

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

function getOpenAIClient(apiKey?: string): OpenAI {
  if (!openaiClient || openaiClientApiKey != apiKey) {
    openaiClient = new OpenAI(apiKey ? { apiKey } : undefined);
    openaiClientApiKey = apiKey;
  }
  return openaiClient;
}

function getEmbeddingConfig(): EmbeddingConfig {
  const config = loadConfig();
  const provider = (config?.embedding_provider || 'openai') as EmbeddingProvider;

  if (provider === 'minimax') {
    return {
      provider,
      model: config?.embedding_model || MINIMAX_MODEL,
      dimensions: config?.embedding_dimensions || DEFAULT_EMBEDDING_DIMENSIONS,
      apiKey: config?.minimax_api_key,
      groupId: config?.minimax_group_id,
      baseUrl: config?.minimax_base_url || config?.embedding_base_url || 'https://api.minimax.chat/v1',
    };
  }

  return {
    provider: 'openai',
    model: config?.embedding_model || OPENAI_MODEL,
    dimensions: config?.embedding_dimensions || DEFAULT_EMBEDDING_DIMENSIONS,
    apiKey: config?.openai_api_key,
    baseUrl: config?.embedding_base_url,
  };
}

export function getEmbeddingProvider(): EmbeddingProvider {
  return getEmbeddingConfig().provider;
}

export function getEmbeddingModel(): string {
  return getEmbeddingConfig().model;
}

export function getEmbeddingDimensions(): number {
  return getEmbeddingConfig().dimensions;
}

export function hasEmbeddingProviderCredentials(): boolean {
  const config = getEmbeddingConfig();
  if (config.provider === 'minimax') {
    return Boolean(config.apiKey && config.groupId);
  }
  return Boolean(config.apiKey);
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
      if (cfg?.apiKey) {
        if (cfg.provider === 'minimax') {
          return await embedWithMinimax(cfg, texts, kind);
        }

        const response = await getClient().embeddings.create({
          model: cfg.model,
          input: texts,
          ...(cfg.dimensions && cfg.provider !== 'minimax' ? { dimensions: cfg.dimensions } : {}),
        });

        const sorted = response.data.sort((a, b) => a.index - b.index);
        return sorted.map(d => new Float32Array(d.embedding));
      }

      const config = getEmbeddingConfig();
      if (!config.apiKey) {
        throw new Error(`Missing API key for embedding provider: ${config.provider}`);
      }

      if (config.provider === 'minimax') {
        return await createMiniMaxEmbeddings(texts, kind, config);
      }
      return await createOpenAIEmbeddings(texts, config);
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

async function createOpenAIEmbeddings(texts: string[], config: EmbeddingConfig): Promise<Float32Array[]> {
  const client = config.baseUrl
    ? new OpenAI({ apiKey: config.apiKey, baseURL: config.baseUrl })
    : getOpenAIClient(config.apiKey);
  const response = await client.embeddings.create({
    model: config.model,
    input: texts,
    dimensions: config.dimensions,
  });

  const sorted = response.data.sort((a, b) => a.index - b.index);
  return sorted.map(d => new Float32Array(d.embedding));
}

async function createMiniMaxEmbeddings(texts: string[], kind: EmbeddingKind, config: EmbeddingConfig): Promise<Float32Array[]> {
  const cfg: EmbeddingProviderConfig = {
    provider: 'minimax',
    apiKey: config.apiKey || '',
    model: config.model,
    baseURL: config.baseUrl,
    dimensions: config.dimensions,
    groupId: config.groupId,
  } as EmbeddingProviderConfig & { groupId?: string };
  return embedWithMinimax(cfg, texts, kind);
}

async function embedWithMinimax(
  cfg: EmbeddingProviderConfig,
  texts: string[],
  kind: EmbeddingKind,
): Promise<Float32Array[]> {
  await waitForMinimaxRequestSlot();

  const baseURL = (cfg.baseURL || 'https://api.minimax.chat/v1').replace(/\/$/, '');
  const groupId = (cfg as EmbeddingProviderConfig & { groupId?: string }).groupId;
  const url = groupId ? `${baseURL}/embeddings?GroupId=${encodeURIComponent(groupId)}` : `${baseURL}/embeddings`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: cfg.apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: cfg.model,
      texts,
      type: kind === 'query' ? 'query' : 'db',
      ...(cfg.dimensions && cfg.provider !== 'minimax' ? { dimensions: cfg.dimensions } : {}),
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

function getBatchSize(): number {
  const configured = parseInt(process.env.GBRAIN_EMBED_BATCH_SIZE || '', 10);
  if (Number.isFinite(configured) && configured > 0) {
    return configured;
  }

  const provider = loadEmbeddingProviderConfig()?.provider || getEmbeddingConfig().provider;
  return provider === 'minimax' ? 1 : DEFAULT_BATCH_SIZE;
}

function exponentialDelay(attempt: number): number {
  return Math.min(BASE_DELAY_MS * (2 ** attempt), MAX_DELAY_MS);
}

function getMinimaxRetryAfterMs(headers: Headers): number {
  const retryAfter = headers.get('retry-after');
  if (retryAfter) {
    const parsed = parseInt(retryAfter, 10);
    if (!isNaN(parsed) && parsed > 0) {
      return parsed * 1000;
    }
  }
  return DEFAULT_MINIMAX_REQUEST_INTERVAL_MS;
}

async function waitForMinimaxRequestSlot(): Promise<void> {
  const now = Date.now();
  const delay = minimaxNextRequestAt - now;
  if (delay > 0) {
    await sleep(delay);
  }
  minimaxNextRequestAt = Date.now() + DEFAULT_MINIMAX_REQUEST_INTERVAL_MS;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export const EMBEDDING_MODEL = () => getEmbeddingConfig().model;
export const EMBEDDING_DIMENSIONS = () => getEmbeddingConfig().dimensions;
export { hasEmbeddingProvider };
