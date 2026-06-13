import Koa from 'koa';
import Router from 'koa-router';
import cors from '@koa/cors';
import bodyParser from 'koa-bodyparser';
import multer from '@koa/multer';
import { createWorker } from 'tesseract.js';
import sharp from 'sharp';

const app = new Koa();
const router = new Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 4 * 1024 * 1024
  }
});

let workerPromise;
let workerInstance = null;
const OCR_TIMEOUT_MS = 45_000;

function cleanOcrText(text) {
  return String(text || '')
    .replace(/\r/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

async function getWorker() {
  if (!workerPromise) {
    workerPromise = (async () => {
      const worker = await createWorker('chi_sim+eng', 1, {
        logger: () => {}
      });
      await worker.setParameters({
        preserve_interword_spaces: '1',
        tessedit_pageseg_mode: '6',
        tessedit_char_blacklist: '|'
      });
      workerInstance = worker;
      return worker;
    })();
  }
  return workerPromise;
}

async function normalizeImage(buffer) {
  return sharp(buffer, { limitInputPixels: 12_000_000 })
    .rotate()
    .resize({
      width: 1600,
      height: 1600,
      fit: 'inside',
      withoutEnlargement: true
    })
    .grayscale()
    .normalize()
    .jpeg({ quality: 82, mozjpeg: true })
    .toBuffer();
}

async function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('ocr_timeout')), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function recognize(buffer) {
  const result = await withTimeout((async () => {
    const input = await normalizeImage(buffer);
    const worker = await getWorker();
    return worker.recognize(input);
  })(), OCR_TIMEOUT_MS);
  const rawText = result?.data?.text || '';
  const text = cleanOcrText(rawText);
  const lines = (result?.data?.lines || [])
    .map((line) => cleanOcrText(line.text))
    .filter(Boolean);

  return {
    text,
    lines,
    words_result: lines.map((words) => ({ words }))
  };
}

router.get('/', (ctx) => {
  ctx.body = {
    ok: true,
    service: 'wokedu-ocr-service',
    endpoints: ['/api/ocr/recognize', '/ocr/recognize', '/vision/ocr', '/ai/ocr']
  };
});

router.get('/health', (ctx) => {
  ctx.body = { ok: true };
});

async function handleOcr(ctx) {
  const file = ctx.file;
  if (!file?.buffer) {
    ctx.status = 400;
    ctx.body = {
      ok: false,
      error: 'missing_file',
      message: 'Please upload an image file with multipart/form-data field "file".'
    };
    return;
  }

  try {
    const data = await recognize(file.buffer);
    ctx.body = {
      ok: true,
      ...data,
      data
    };
  } catch (error) {
    if (error?.message === 'ocr_timeout') {
      if (workerInstance) {
        workerInstance.terminate().catch(() => {});
      }
      workerInstance = null;
      workerPromise = null;
      ctx.body = {
        ok: true,
        text: '',
        lines: [],
        words_result: [],
        warning: 'ocr_timeout',
        message: '图片识别耗时较长，请换一张更清晰的横向试题照片重试。',
        data: {
          text: '',
          lines: [],
          words_result: []
        }
      };
      return;
    }

    ctx.status = 500;
    ctx.body = {
      ok: false,
      error: 'ocr_failed',
      message: error?.message || 'OCR failed.'
    };
  }
}

router.post('/api/ocr/recognize', upload.single('file'), handleOcr);
router.post('/ocr/recognize', upload.single('file'), handleOcr);
router.post('/vision/ocr', upload.single('file'), handleOcr);
router.post('/ai/ocr', upload.single('file'), handleOcr);

app.use(cors());
app.use(bodyParser());
app.use(router.routes());
app.use(router.allowedMethods());

const port = Number(process.env.BYTEFAAS_RUNTIME_PORT || 8000);
app.listen(port, '0.0.0.0', () => {
  console.log(`wokedu OCR service listening on ${port}`);
});
