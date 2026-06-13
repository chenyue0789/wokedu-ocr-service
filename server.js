import Koa from 'koa';
import Router from 'koa-router';
import cors from '@koa/cors';
import bodyParser from 'koa-bodyparser';
import multer from '@koa/multer';
import { createWorker } from 'tesseract.js';

const app = new Koa();
const router = new Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 8 * 1024 * 1024
  }
});

let workerPromise;

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
        tessedit_pageseg_mode: '6'
      });
      return worker;
    })();
  }
  return workerPromise;
}

async function recognize(buffer) {
  const worker = await getWorker();
  const result = await worker.recognize(buffer);
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

const port = Number(process.env.PORT || 8080);
app.listen(port, '0.0.0.0', () => {
  console.log(`wokedu OCR service listening on ${port}`);
});
