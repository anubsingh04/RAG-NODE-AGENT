import { logger} from "@godspeedsystems/core";
import fetch from "node-fetch";
import * as fs from 'fs/promises';
import { VectorStore } from './vectorStore';
import pdfParse from 'pdf-parse';
import { createWorker } from 'tesseract.js';
import * as path from 'path';
import mammoth from "mammoth";
import { parse as htmlParse } from "node-html-parser";

const vs=new VectorStore();

interface GitTreeItem {
  path: string;
  type: string;
}

interface GITHUBCOMMIT{
  owner?: string ,
  repo?: string ,
  branch?: string ,
  commit?: string ,

}

interface GitTreeResponse {
  tree: GitTreeItem[];
}
interface GitHubTree {
  tree: { path: string; type: string }[];
}
interface CommitResponse {
  sha: string;
}
interface FileChange {
  filename: string;
  status: 'added' | 'modified' | 'removed';
}

interface CompareResponse {
  files: FileChange[];
}

const COMMIT_FILE = path.resolve(__dirname, '../../data/last_commit.json');
const REPO_URL_FILE = path.resolve(__dirname, '../../data/repo_url.json');


async function saveLastCommit(owner: string, repo: string, branch: string, commitSha: string) {
  let parsed: GITHUBCOMMIT[] = [];

  try {
    const data = await fs.readFile(COMMIT_FILE, 'utf-8');
    parsed = JSON.parse(data);
  } catch (err: any) {
    if (err.code !== 'ENOENT') {
      logger.error('Unexpected error while reading commit file:', err);
    }
  }

  const index = parsed.findIndex((p) => p.owner === owner && p.repo === repo && p.branch === branch);
  if (index !== -1) {
    parsed[index].commit = commitSha;
  } else {
    parsed.push({ owner, repo, branch, commit: commitSha });
  }

  await fs.writeFile(COMMIT_FILE, JSON.stringify(parsed, null, 2), 'utf-8');
}


async function loadLastCommit(owner: string, repo: string, branch: string) {
    let parsed: GITHUBCOMMIT[] = [];  
    try {
        const data = await fs.readFile(COMMIT_FILE, 'utf-8');
        parsed = JSON.parse(data);
        const index = parsed.findIndex((p: GITHUBCOMMIT)  => p.owner === owner && p.repo === repo && p.branch === branch);
        if (index !== -1) {
            const finobj = parsed[index] 
            return {repo: finobj.repo, commit: finobj.commit}
        } 
        return {}
    } catch {
        return {};
    }
}

async function loadRepoUrl() {
    try {
        const data = await fs.readFile(REPO_URL_FILE, 'utf-8');
        
        return JSON.parse(data);
    }
    catch {
        return {};
    }
}

async function getLatestCommitSha(owner: string, repo: string, branch: string): Promise<string> {
    const url = `https://api.github.com/repos/${owner}/${repo}/commits/${branch}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch latest commit SHA: ${res.statusText}`);
    const json = await res.json() as CommitResponse;
    return json.sha;
}

async function getChangedFiles(owner: string, repo: string, baseSha: string, headSha: string) {
    const url = `https://api.github.com/repos/${owner}/${repo}/compare/${baseSha}...${headSha}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch changed files: ${res.statusText}`);
    const json = await res.json() as CompareResponse;
    const changed: string[] = [];
    const deleted: string[] = [];

    for (const file of json.files || []) {
        if (file.status === 'modified' || file.status === 'added') changed.push(file.filename);
        else if (file.status === 'removed') deleted.push(file.filename);
    }
    return { changed, deleted };
}

async function extractTextFromPdf(buffer: Buffer): Promise<string> {
    try {
        const data = await pdfParse(buffer);
        if (data.text && data.text.trim().length > 30) {
            return data.text;
        } else {
            const worker = await createWorker();
            const w = await worker as any;
            await w.load();
            await w.loadLanguage('eng');
            await w.initialize('eng');
            const { data: { text } } = await w.recognize(buffer);
            await w.terminate();
            return text;
        }
    } catch (e) {
        return '';
    }
}
async function ingestChangedFiles(repoUrl: string, branch = 'main'): Promise<void> {
    const parts = repoUrl.replace(/\/$/, '').split('/');
    const owner = parts[parts.length - 2];
    const repo = parts[parts.length - 1];
    const latestSha = await getLatestCommitSha(owner, repo, branch);
    const state = await loadLastCommit(owner, repo, branch);
    if (state.repo === repo && state.commit === latestSha) {
        logger.info('No new commit. Skipping ingestion.');
        return;
    }
    // const vs = new VectorStore();
    let changedFiles: string[] = [];
    let deletedFiles: string[] = [];
    if (state.repo === repo && state.commit) {
        logger.info("Getting changed files")
        const changes = await getChangedFiles(owner, repo, state.commit, latestSha);
        changedFiles = changes.changed;
        deletedFiles = changes.deleted;
    }
    else {
        // First time: get all files from HEAD
        const treeUrl = `https://api.github.com/repos/${owner}/${repo}/git/trees/${branch}?recursive=1`;
        const treeRes = await fetch(treeUrl);
        if (!treeRes.ok)
            throw new Error(`Failed to fetch repo tree: ${treeRes.statusText}`);
        const treeJson = (await treeRes.json()) as GitHubTree;
        changedFiles = treeJson.tree
            .filter((f: any) => f.type === 'blob')
            .map((f: any) => f.path);
        deletedFiles = [];
    }
    for (const filePath of deletedFiles) {
        logger.info(`Removing deleted file from vector DB: ${filePath}`);
        await vs.removeDocument(filePath);
    }
    const allowedExts = new Set(['.md', '.txt', '.pdf', '.mdx']);
    for (const filePath of changedFiles) {
        try {
            const ext = path.extname(filePath).toLowerCase();
            if (!allowedExts.has(ext))
                continue;
            const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${filePath}`;
            const contentRes = await fetch(rawUrl);
            if (!contentRes.ok) {
                logger.error(`Failed to fetch content for: ${filePath} (status ${contentRes.status})`);
                continue;
            }
            let content;
            if (ext === '.pdf') {
                const buffer = await contentRes.buffer();
                content = await extractTextFromPdf(buffer);
            }
            else {
                content = await contentRes.text();
            }
            if (content.length > 0) {
                logger.info(`Re-ingesting file: ${filePath}`);
                await vs.removeDocument(filePath);
                await vs.upsert(filePath, content);

            }
        }
        catch (e) {
            logger.error(`Error processing file ${filePath}:`, e);
        }
    }
    await saveLastCommit(owner, repo, branch , latestSha);
    logger.info('Ingestion complete.');
}

async function ingestUploadedFile(
  file: Buffer,
  filename: string,
  docUniqueId: string,
  vs: VectorStore
): Promise<string> {
  const ext = path.extname(filename).toLowerCase();
  const buffer = file;

  let content = "";

  switch (ext) {
    case ".pdf":
      const pdf = await extractTextFromPdf(buffer);
      content = pdf;
      break;

    case ".docx":
      const result = await mammoth.extractRawText({ buffer });
      content = result.value;
      break;

    case ".txt":
    case ".md":
      content = buffer.toString("utf-8");
      break;

    case ".html":
      const root = htmlParse(buffer.toString("utf-8"));
      content = root.text;
      break;

    default:
      return `Unsupported file type: ${ext}`;
  }

  const fname = path.basename(filename, path.extname(filename));
  const docId = docUniqueId;
  logger.info(`[${fname} with docID ${docId}] Starting ingestion.`);

  // Step 1: Split by pages or sections
  const pages = content.split(/\n{2,}/).filter((p) => p.trim().length > 0);
  logger.info(`[${fname} with docID ${docId}] ${pages.length} logical pages found.`);

  // Step 2: Process each page
  for (let i = 0; i < pages.length; i++) {
    const pageContent = pages[i];
    const pageId = `${docId}_page_${i + 1}`;
    await vs.upsert(pageId, pageContent);
  }
  logger.info(`Document '${filename}' ingested successfully.`);
  return `Document '${filename}' ingested successfully.`;
}

export { ingestChangedFiles, loadRepoUrl, ingestUploadedFile}
