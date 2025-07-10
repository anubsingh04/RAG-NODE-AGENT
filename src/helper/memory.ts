import { BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { GSContext } from '@godspeedsystems/core';

export class RedisCheckpointSaver extends BaseCheckpointSaver {
  private ctx: GSContext;

  constructor(ctx: GSContext) {
    super();
    this.ctx = ctx;
  }

  async getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined> {
    const thread_id = config.configurable?.thread_id;
    if (!thread_id) return undefined;

    try {
      const key = `checkpoint:${thread_id}`;
      const result = await this.ctx.datasources.redis.get({ key });
      
      if (!result || !result.data) return undefined;
      
      const data = JSON.parse(result.data);
      return {
        config,
        checkpoint: data.checkpoint,
        metadata: data.metadata || {},
        parentConfig: data.parentConfig
      };
    } catch (error) {
      this.ctx.logger.error(`Error retrieving checkpoint: ${error}`);
      return undefined;
    }
  }

  async put(config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata): Promise<RunnableConfig> {
    const thread_id = config.configurable?.thread_id;
    if (!thread_id) throw new Error("thread_id is required");

    try {
      const key = `checkpoint:${thread_id}`;
      const value = JSON.stringify({
        checkpoint,
        metadata,
        parentConfig: config
      });
      
      await this.ctx.datasources.redis.set({ key, value });
      return config;
    } catch (error) {
      this.ctx.logger.error(`Error saving checkpoint: ${error}`);
      throw error;
    }
  }

  async putWrites(config: RunnableConfig, writes: any[], taskId: string): Promise<void> {
    // For simplicity, we'll store writes as part of the checkpoint
    // In a more sophisticated implementation, you might store writes separately
    const thread_id = config.configurable?.thread_id;
    if (!thread_id) return;

    try {
      const key = `writes:${thread_id}:${taskId}`;
      const value = JSON.stringify(writes);
      await this.ctx.datasources.redis.set({ key, value });
    } catch (error) {
      this.ctx.logger.error(`Error saving writes: ${error}`);
    }
  }

  async *list(config: RunnableConfig, options?: any): AsyncGenerator<CheckpointTuple> {
    // For simplicity, we'll just return the current checkpoint if it exists
    const tuple = await this.getTuple(config);
    if (tuple) {
      yield tuple;
    }
  }
}

export function createRedisMemorySaver(ctx: GSContext): RedisCheckpointSaver {
  return new RedisCheckpointSaver(ctx);
}
