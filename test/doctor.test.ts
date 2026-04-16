import { describe, test, expect, mock } from 'bun:test';

mock.module('../src/core/embedding.ts', () => ({
  getEmbeddingProvider: () => 'minimax',
  hasEmbeddingProviderCredentials: () => false,
}));

describe('doctor command', () => {
  test('doctor module exports runDoctor', async () => {
    const { runDoctor } = await import('../src/commands/doctor.ts');
    expect(typeof runDoctor).toBe('function');
  });

  test('LATEST_VERSION is importable from migrate', async () => {
    const { LATEST_VERSION } = await import('../src/core/migrate.ts');
    expect(typeof LATEST_VERSION).toBe('number');
  });

  test('doctor warns when configured embedding provider has no credentials', async () => {
    const { runDoctor } = await import('../src/commands/doctor.ts');

    const lines: string[] = [];
    const originalLog = console.log;
    const originalExit = process.exit;

    console.log = (...args: any[]) => { lines.push(args.join(' ')); };
    (process.exit as any) = ((code?: number) => { throw new Error(`EXIT:${code ?? 0}`); }) as any;

    const engine = {
      getStats: async () => ({ page_count: 1 }),
      getConfig: async () => '1',
      getHealth: async () => ({ embed_coverage: 0, missing_embeddings: 1 }),
    } as any;

    try {
      await runDoctor(engine, []);
    } catch (e: any) {
      expect(String(e.message)).toContain('EXIT:0');
    } finally {
      console.log = originalLog;
      process.exit = originalExit;
    }

    expect(lines.join('\n')).toContain('Embedding provider "minimax" is configured without credentials');
    expect(lines.join('\n')).not.toContain('embed refresh');
  });
});
