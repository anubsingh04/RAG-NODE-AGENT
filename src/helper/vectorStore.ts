import { ChromaClient, IncludeEnum } from "chromadb";
import { GoogleGeminiEmbeddingFunction } from "@chroma-core/google-gemini";
import * as fs from 'fs';
import * as path from 'path';
import crypto from 'crypto';
import pLimit from 'p-limit';

interface Metadata {
  [docId: string]: {
    content: string;
    documentType?: string;
    createdAt?: string;
    lastModified?: string;
  };
}

interface ChunkMetadata {
  id: string;
  content: string;
  startIndex: number;
  endIndex: number;
  chunkType: 'paragraph' | 'section' | 'code' | 'list';
}

interface SearchResult {
  docId: string;
  chunkId: string;
  chunkContent: string;
  metadata: {
    documentType?: string;
    chunkType: string;
  };
  relevanceScore: number;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const chromaLimit = pLimit(3); // max 3 concurrent ChromaDB calls

async function safeChroma<T>(fn: () => Promise<T>, retries = 3): Promise<T> {
  for (let i = 0; i < retries; i++) {
    try {
      return await chromaLimit(fn);
    } catch (err: any) {
      const isRecoverable =
        err?.message?.includes("ECONNREFUSED") ||
        err?.message?.includes("Failed to connect to chromadb") ||
        err?.message?.includes("connection") ||
        err?.message?.includes("fetch failed");

      if (isRecoverable && i < retries - 1) {
        const wait = 1000 * Math.pow(2, i);
        console.warn(`[ChromaDB] Retrying in ${wait / 1000}s...`);
        await sleep(wait);
      } else {
        throw err;
      }
    }
  }
  throw new Error("ChromaDB request failed after retries");
}

export class VectorStore {
  private metaPath: string;
  private chunkMapPath: string;
  public metadata: Metadata = {};
  private chunkMap: Record<string, ChunkMetadata[]> = {};
  private client: ChromaClient;
  private collectionName = 'rag-collection';
  private embedder: GoogleGeminiEmbeddingFunction;

  constructor(
    metaPath = path.resolve(__dirname, '../../index/metadata.json'),
    chunkMapPath = path.resolve(__dirname, '../../index/chunkmap.json')
  ) {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) throw new Error('Missing GOOGLE_API_KEY in .env');

    this.metaPath = metaPath;
    this.chunkMapPath = chunkMapPath;

    this.embedder = new GoogleGeminiEmbeddingFunction({ apiKey });
    this.client = new ChromaClient({ host: "localhost", port: 10947, ssl: false });

    this.load();
  }

  private load(): void {
    if (fs.existsSync(this.metaPath)) {
      try {
        this.metadata = JSON.parse(fs.readFileSync(this.metaPath, 'utf-8'));
      } catch (e) {
        console.error('[ERROR] Failed to load metadata:', e);
      }
    }

    if (fs.existsSync(this.chunkMapPath)) {
      try {
        this.chunkMap = JSON.parse(fs.readFileSync(this.chunkMapPath, 'utf-8'));
      } catch (e) {
        console.error('[ERROR] Failed to load chunkMap:', e);
      }
    }
  }

  private ensureDir(filePath: string) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }

  private save(): void {
    try {
      this.ensureDir(this.metaPath);
      fs.writeFileSync(this.metaPath, JSON.stringify(this.metadata, null, 2));
      fs.writeFileSync(this.chunkMapPath, JSON.stringify(this.chunkMap, null, 2));
    } catch (err) {
      console.error('Failed to write metadata or chunkMap:', err);
    }
  }

  private semanticChunkText(text: string): ChunkMetadata[] {
    const chunks: ChunkMetadata[] = [];
    const paragraphs = text.split(/\n\s*\n/);
    let cursor = 0;

    for (const paragraph of paragraphs) {
      const trimmed = paragraph.trim();
      if (trimmed.length === 0) continue;

      const chunkType = paragraph.includes('```') || /function|class/.test(paragraph) ? 'code'
        : /^\s*[-*+]\s|^\s*\d+\.\s/.test(paragraph) ? 'list'
        : /^#{1,6}\s|^[A-Z][A-Z\s]+$/.test(paragraph) ? 'section'
        : 'paragraph';

      if (trimmed.length > 1000 && chunkType === 'paragraph') {
        const sentences = trimmed.split(/[.!?]+/);
        let currentChunk = '';
        for (const sentence of sentences) {
          if ((currentChunk + sentence).length > 1000 && currentChunk.length > 0) {
            chunks.push({
              id: crypto.randomUUID(),
              content: currentChunk.trim(),
              startIndex: cursor,
              endIndex: cursor + currentChunk.trim().length,
              chunkType: 'paragraph',
            });
            cursor += currentChunk.length;
            currentChunk = sentence;
          } else {
            currentChunk += sentence + '. ';
          }
        }
        if (currentChunk.trim().length > 0) {
          chunks.push({
            id: crypto.randomUUID(),
            content: currentChunk.trim(),
            startIndex: cursor,
            endIndex: cursor + currentChunk.trim().length,
            chunkType: 'paragraph',
          });
          cursor += currentChunk.length;
        }
      } else {
        chunks.push({
          id: crypto.randomUUID(),
          content: trimmed,
          startIndex: cursor,
          endIndex: cursor + trimmed.length,
          chunkType,
        });
        cursor += paragraph.length + 2;
      }
    }

    return chunks;
  }

  async upsert(docId: string, content: string, documentType?: string): Promise<void> {
    const chunks = this.semanticChunkText(content);
    const collection = await safeChroma(() =>
      this.client.getOrCreateCollection({
        name: this.collectionName,
        embeddingFunction: this.embedder,
      })
    );

    const BATCH_SIZE = 100;
    const DELAY_MS = 500;

    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const batch = chunks.slice(i, i + BATCH_SIZE);

      let retries = 0;
      let success = false;

      while (!success && retries < 3) {
        try {
          await safeChroma(() =>
            collection.upsert({
              ids: batch.map(chunk => chunk.id),
              documents: batch.map(chunk => chunk.content),
              metadatas: batch.map(chunk => {
                const base: Record<string, string> = {
                  docId,
                  chunkType: chunk.chunkType,
                };
                return documentType ? { ...base, documentType } : base;
              }),
            })
          );
          success = true;
        } catch (err) {
          console.error(`[Gemini/Chroma] Upsert error:`, err);
          const wait = 2000 * (retries + 1);
          await sleep(wait);
          retries++;
        }
      }

      if (success && i + BATCH_SIZE < chunks.length) {
        await sleep(DELAY_MS);
      }
    }

    this.metadata[docId] = {
      content,
      documentType,
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString(),
    };
    this.chunkMap[docId] = chunks;
    this.save();
  }

  async removeDocument(docId: string) {
    if (!(docId in this.metadata)) return;

    const collection = await safeChroma(() =>
      this.client.getOrCreateCollection({
        name: this.collectionName,
        embeddingFunction: this.embedder,
      })
    );

    const chunkIds = (this.chunkMap[docId] || []).map(chunk => chunk.id);
    if (chunkIds.length > 0) {
      await safeChroma(() => collection.delete({ ids: chunkIds }));
    }

    delete this.metadata[docId];
    delete this.chunkMap[docId];
    this.save();
  }

  async search(query: string, k = 10, filters?: { documentType?: string; chunkType?: string }): Promise<SearchResult[]> {
    const collection = await safeChroma(() =>
      this.client.getCollection({
        name: this.collectionName,
        embeddingFunction: this.embedder,
      })
    );

    const filterConditions: Record<string, any>[] = [];
    if (filters?.documentType) filterConditions.push({ documentType: filters.documentType });
    if (filters?.chunkType) filterConditions.push({ chunkType: filters.chunkType });
    const where = filterConditions.length > 0 ? { $and: filterConditions } : undefined;

    const results = await safeChroma(() =>
      collection.query({
        queryTexts: [query],
        nResults: k,
        where,
        include: ["documents", "metadatas", "distances"] as IncludeEnum[],
      })
    );

    const output: SearchResult[] = [];
    for (let i = 0; i < results.ids[0].length; i++) {
      output.push({
        docId: String(results.metadatas?.[0][i]?.docId ?? ''),
        chunkId: results.ids[0][i],
        chunkContent: String(results.documents?.[0][i] ?? ''),
        metadata: {
          documentType: String(results.metadatas?.[0][i]?.documentType ?? ''),
          chunkType: String(results.metadatas?.[0][i]?.chunkType ?? ''),
        },
        relevanceScore: results.distances?.[0][i] ?? 0,
      });
    }
    return output;
  }

  getChunks(docId: string): ChunkMetadata[] {
    return this.chunkMap[docId] || [];
  }
}
