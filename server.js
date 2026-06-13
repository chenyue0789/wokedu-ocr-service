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
const OCR_TIMEOUT_MS = 50_000;
const OCR_SPACE_ENDPOINT = 'https://api.ocr.space/parse/image';

function cleanOcrText(text) {
  return String(text || '')
    .replace(/\r/g, '')
    .replace(/[^\u4e00-\u9fa5a-zA-Z0-9+\-xX×÷=<>≤≥().,，。:：;；?？!！\[\]（）\s]/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/[ \t]{2,}/g, ' ')
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
        tessedit_char_blacklist: '|'
      });
      workerInstance = worker;
      return worker;
    })();
  }
  return workerPromise;
}

async function buildImageVariants(buffer) {
  const base = sharp(buffer, { limitInputPixels: 12_000_000 })
    .rotate()
    .resize({
      width: 2200,
      height: 2200,
      fit: 'inside',
      withoutEnlargement: true
    })
    .grayscale()
    .normalize()
    .sharpen();

  const enhanced = await base
    .clone()
    .threshold(170)
    .jpeg({ quality: 90, mozjpeg: true })
    .toBuffer();

  const gray = await base
    .clone()
    .jpeg({ quality: 86, mozjpeg: true })
    .toBuffer();

  return [
    { name: 'enhanced_sparse', psm: '11', input: enhanced },
    { name: 'gray_block', psm: '6', input: gray }
  ];
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
  if (process.env.OCR_SPACE_API_KEY) {
    try {
      return await withTimeout(recognizeWithOcrSpace(buffer), 20_000);
    } catch (error) {
      console.warn('ocr.space failed, fallback to local OCR:', error?.message || error);
    }
  }

  const result = await withTimeout((async () => {
    const variants = await buildImageVariants(buffer);
    const worker = await getWorker();
    let best = null;

    for (const variant of variants) {
      await worker.setParameters({
        preserve_interword_spaces: '1',
        tessedit_pageseg_mode: variant.psm,
        tessedit_char_blacklist: '|'
      });
      const result = await worker.recognize(variant.input);
      const text = cleanOcrText(result?.data?.text || '');
      const score = scoreOcrText(text);

      if (!best || score > best.score) {
        best = { result, text, score, variant: variant.name };
      }

      if (score >= 18) {
        break;
      }
    }

    return best;
  })(), OCR_TIMEOUT_MS);
  const rawText = result?.result?.data?.text || '';
  const text = result?.text || cleanOcrText(rawText);
  const lines = (result?.result?.data?.lines || [])
    .map((line) => cleanOcrText(line.text))
    .filter(Boolean);

  return {
    text,
    lines,
    words_result: lines.map((words) => ({ words })),
    confidence: result?.result?.data?.confidence || 0,
    ocr_variant: result?.variant || 'unknown',
    ocr_provider: 'tesseract'
  };
}

async function recognizeWithOcrSpace(buffer) {
  const form = new FormData();
  form.append('apikey', process.env.OCR_SPACE_API_KEY);
  form.append('language', 'chs');
  form.append('OCREngine', '2');
  form.append('scale', 'true');
  form.append('isTable', 'true');
  form.append('file', new Blob([buffer], { type: 'image/jpeg' }), 'question.jpg');

  const response = await fetch(OCR_SPACE_ENDPOINT, {
    method: 'POST',
    body: form
  });

  if (!response.ok) {
    throw new Error(`ocr_space_http_${response.status}`);
  }

  const payload = await response.json();
  if (payload?.IsErroredOnProcessing) {
    throw new Error(payload?.ErrorMessage?.[0] || payload?.ErrorDetails || 'ocr_space_error');
  }

  const parsed = payload?.ParsedResults?.[0];
  const text = cleanOcrText(parsed?.ParsedText || '');
  const lines = text.split(/\n+/).map((line) => cleanOcrText(line)).filter(Boolean);

  return {
    text,
    lines,
    words_result: lines.map((words) => ({ words })),
    confidence: 80,
    ocr_variant: 'ocr_space_engine_2',
    ocr_provider: 'ocr.space'
  };
}

function scoreOcrText(text) {
  const chinese = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const digits = (text.match(/[0-9]/g) || []).length;
  const math = (text.match(/[+\-xX×÷=<>≤≥]/g) || []).length;
  const educationWords = (text.match(/数学|语文|英语|题|卷|年级|计算|选择|填空|阅读|作文/g) || []).length * 4;
  const latinNoise = (text.match(/[a-zA-Z]{4,}/g) || []).length * 2;
  return chinese + Math.min(digits, 20) + math + educationWords - latinNoise;
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
