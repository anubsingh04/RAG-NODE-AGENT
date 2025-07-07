import { VectorStore } from './vectorStore';
import OpenAI from 'openai';
import * as fs from 'fs';
import * as path from 'path';

const logStream = fs.createWriteStream('logs.txt', { flags: 'a' });

const log = function (message: any) {
  logStream.write(`${new Date().toISOString()} - ${message}\n`);
};

interface QueryContext {
  context: string;
  source_files: string;
  // relevance_scores: number[];
  // chunk_types: string[];
  // total_chunks: number;
}

export class RAGPipeline {
  private vs: VectorStore;

  constructor() {
    log('Loading vector store...');
    this.vs = new VectorStore();
    log('Vector store loaded...');
  }

  async run(
    query: string,
    options: {
      maxResults?: number;
      minRelevanceScore?: number;
      includeMetadata?: boolean;
      filterByChunkType?: string[];
      maxContextLength?: number;
      includeFullDocuments?: boolean;
    } = {},
  ): Promise<QueryContext> {
    const {
      maxResults = 5,
      minRelevanceScore = 0.3,
      includeMetadata = false,
      filterByChunkType = [],
      maxContextLength = 4000,
      includeFullDocuments = true,
    } = options;

    log('Processing query: ' + query);
    const optimizedQuery = this.optimizeQuery(query);
    log('Optimized query: ' + optimizedQuery);

    const filters: any = { minRelevanceScore };
    let docs: any[] = [];

    if (filterByChunkType.length > 0) {
      const allResults: any[] = [];
      for (const chunkType of filterByChunkType) {
        const results = await this.vs.search(
          optimizedQuery,
          Math.ceil(maxResults / filterByChunkType.length),
          { ...filters, chunkType },
        );
        allResults.push(...results);
      }
      allResults.sort((a: any, b: any) => b.relevanceScore - a.relevanceScore);
      docs = allResults.slice(0, maxResults);
    } else {
      docs = await this.vs.search(optimizedQuery, maxResults , filters);
    }

    log(`Found ${docs.length} relevant documents`);

    const uniqueDocIds = Array.from(new Set(docs.map((doc) => doc.docId)));
    const sourceFiles = uniqueDocIds.join('\n');

    let context = '';
    let relevanceScores: number[] = [];
    let chunkTypes: string[] = [];
    let totalChunks = 0;

    if (includeFullDocuments) {
      const fullDocuments = uniqueDocIds
        .map((docId) => {
          const meta = this.vs.metadata[docId];
          if (!meta) {
            log(`[WARN] No metadata found for ${docId}`);
            return null;
          }

          let docHeader = `[File: ${docId}]`;
          // if (meta.documentType) {
          //   docHeader += ` [Type: ${meta.documentType}]`;
          // }

          return `${docHeader}\n\n${meta.content}`;
        })
        .filter(Boolean);


      context = fullDocuments.join('\n\n');
      relevanceScores = docs.map((d) => d.relevanceScore);
      chunkTypes = docs.map((d) => d.metadata.chunkType);
      totalChunks = uniqueDocIds.length;
    } else {
      const selectedChunks = this.selectBestChunks(docs, maxContextLength);
      const uniqueChunks = this.deduplicateChunks(selectedChunks);
      context = this.formatContext(uniqueChunks, includeMetadata);
      relevanceScores = selectedChunks.map((c) => c.relevanceScore);
      chunkTypes = selectedChunks.map((c) => c.metadata.chunkType);
      totalChunks = uniqueChunks.length;
    }

    log(
      `Selected ${totalChunks} ${includeFullDocuments ? 'documents' : 'chunks'} from ${uniqueDocIds.length} files`,
    );

    log(`Context length: ${context.length} characters`);
    return {
      context,
      source_files: sourceFiles,
      // relevance_scores: relevanceScores,
      // chunk_types: chunkTypes,
      // total_chunks: totalChunks,
    };
  }

//   async rag(query: string): Promise<{ answer: string; source_files: string[] }> {
        
//     const context = 
//     const openai = new OpenAI({
//             apiKey: process.env.OPENROUTER_API_KEY,
//             baseURL: 'https://openrouter.ai/api/v1',
//           });

//         const response = await openai.chat.completions.create({
//                                        model: process.env.MODEL_NAME!, 
//                                        messages: [
//                                                      {
//                                                       role: 'system',
//                                                       content: "      \
//                                                                 You are a helpful assistant who understands the Godspeed Framework deeply. Always aim to provide technically sound, creative, and helpful answers to a wide range of user questions, using the documentation provided as context. \
//                                                                  \
//                                                                 **Rules:**\
// 1. Always read and understand the full user query and provided context before answering.\
//    - If the answer can be fully derived from the context, then answer with thorough technical clarity using at least 1000 tokens when needed.\
//    - If the answer cannot be fully derived from context, say so sincerely — unless you can add well-grounded insights from general training that logically extend the documentation.\
// \
// 2. Be versatile:\
//    - Explain concepts clearly when asked for definitions or meanings.\
//    - Describe how components work when asked about mechanisms.\
//    - Show how to build new things using given APIs or tools when asked for implementation help.\
// \
// 3. Respond naturally and warmly if the user is just chatting.\
// \
// 4. When including Bash commands:\
//    - Format using fenced bash blocks:\
//      ```bash\
//      # example\
//      godspeed run app.yaml\
//      ```\
// \
// 5. When using math or formulas:\
//    - Always use inline LaTeX: wrap expressions like this — $a^2 + b^2 = c^2$.\
//    - Use $$ for display math on its own line, and always close math blocks properly.\
// \
// Your tone should be friendly but focused. If the user asks something unrelated to the documentation or framework, explain clearly that you are focused on helping with Godspeed-related tasks.".trim()
//       },
//       {
//         role: 'user',
//         content: `Context:\n${context}\n\nQuestion: ${query}\nAnswer:`
//       }
//     ]
// });
        
//         const answer = response.choices?.[0]?.message?.content?.trim() || 'No response.';

//         return {
//             answer: answer,
//             source_files: sourceFiles
//         }
//     }

    private optimizeQuery(query: string): string {
    let optimized = query
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();

    const keyTerms = optimized
      .split(' ')
      .filter((word) => word.length > 3)
      .slice(0, 5);

    return keyTerms.join(' ');
  }

  private selectBestChunks(docs: any[], maxContextLength: number): any[] {
    const selectedChunks: any[] = [];
    let currentLength = 0;

    const sortedDocs = docs.sort(
      (a: any, b: any) => b.relevanceScore - a.relevanceScore,
    );

    for (const doc of sortedDocs) {
      const chunkLength = doc.chunkContent.length;

      if (currentLength + chunkLength > maxContextLength) {
        if (doc.relevanceScore > 0.7) {
          const remainingLength = maxContextLength - currentLength;
          if (remainingLength > 200) {
            const truncatedChunk = {
              ...doc,
              chunkContent:
                doc.chunkContent.substring(0, remainingLength) + '...',
            };
            selectedChunks.push(truncatedChunk);
            currentLength += remainingLength;
          }
        }
        break;
      }

      selectedChunks.push(doc);
      currentLength += chunkLength;
    }

    return selectedChunks;
  }

  private deduplicateChunks(chunks: any[]): any[] {
    const seen = new Set<string>();
    const uniqueChunks: any[] = [];

    for (const chunk of chunks) {
      const contentHash = this.hashContent(chunk.chunkContent);
      if (!seen.has(contentHash)) {
        seen.add(contentHash);
        uniqueChunks.push(chunk);
      }
    }

    return uniqueChunks;
  }

  private hashContent(content: string): string {
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString();
  }

  private formatContext(chunks: any[], includeMetadata: boolean): string {
    const formattedChunks = chunks.map((chunk, index) => {
      let formatted = `[Chunk ${index + 1}] ${chunk.chunkContent}`;

      if (includeMetadata) {
        formatted += `\n[Source: ${chunk.docId}, Type: ${chunk.metadata.chunkType}, Relevance: ${chunk.relevanceScore.toFixed(2)}]`;
      }

      return formatted;
    });

    return formattedChunks.join('\n\n');
  }

  async getContextStats(query: string): Promise<any> {
    const docs = await this.vs.search(query, 20);

    const stats = {
      totalResults: docs.length,
      avgRelevanceScore:
        docs.reduce((sum: number, doc: any) => sum + doc.relevanceScore, 0) /
        docs.length,
      chunkTypeDistribution: {} as Record<string, number>,
      documentTypeDistribution: {} as Record<string, number>,
      topResults: docs.slice(0, 5).map((doc: any) => ({
        docId: doc.docId,
        relevanceScore: doc.relevanceScore,
        chunkType: doc.metadata.chunkType,
        contentPreview: doc.chunkContent.substring(0, 100) + '...',
      })),
    };

    for (const doc of docs) {
      stats.chunkTypeDistribution[doc.metadata.chunkType] =
        (stats.chunkTypeDistribution[doc.metadata.chunkType] || 0) + 1;

      if (doc.metadata.documentType) {
        stats.documentTypeDistribution[doc.metadata.documentType] =
          (stats.documentTypeDistribution[doc.metadata.documentType] || 0) + 1;
      }
    }

    return stats;
  }
}
