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
let baiduTokenCache = null;

const OCR_TIMEOUT_MS = 50_000;
const OCR_SPACE_ENDPOINT = 'https://api.ocr.space/parse/image';
const BAIDU_TOKEN_ENDPOINT = 'https://aip.baidubce.com/oauth/2.0/token';
const BAIDU_PAPER_CUT_ENDPOINT = 'https://aip.baidubce.com/rest/2.0/ocr/v1/paper_cut_edu';
const BAIDU_DOC_ANALYSIS_ENDPOINT = 'https://aip.baidubce.com/rest/2.0/ocr/v1/doc_analysis';

function cleanOcrText(text) {
  return String(text || '')
    .replace(/\r/g, '')
    .replace(/[^\u4e00-\u9fa5a-zA-Z0-9+\-xX×÷=<>≤≥().,，。:：;；?？!！\[\]（）\s]/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function compactLines(lines) {
  return lines
    .map((line) => cleanOcrText(line))
    .filter(Boolean);
}

function makeOcrResponse({ lines, confidence = 0, provider, variant, extra = {} }) {
  const cleanLines = compactLines(lines);
  return {
    text: cleanLines.join('\n'),
    lines: cleanLines,
    words_result: cleanLines.map((words) => ({ words })),
    confidence,
    ocr_variant: variant,
    ocr_provider: provider,
    ...extra
  };
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

function hasBaiduCredentials() {
  return Boolean(
    process.env.BAIDU_ACCESS_TOKEN ||
      ((process.env.BAIDU_OCR_API_KEY || process.env.BAIDU_API_KEY) &&
        (process.env.BAIDU_OCR_SECRET_KEY || process.env.BAIDU_SECRET_KEY))
  );
}

async function recognize(buffer) {
  if (hasBaiduCredentials()) {
    try {
      return await withTimeout(recognizeWithBaidu(buffer), 35_000);
    } catch (error) {
      console.warn('baidu OCR failed, fallback to ocr.space/local:', error?.message || error);
    }
  }

  if (process.env.OCR_SPACE_API_KEY) {
    try {
      return await withTimeout(recognizeWithOcrSpace(buffer), 20_000);
    } catch (error) {
      console.warn('ocr.space failed, fallback to local OCR:', error?.message || error);
    }
  }

  return withTimeout(recognizeWithTesseract(buffer), OCR_TIMEOUT_MS);
}

async function recognizeWithTesseract(buffer) {
  const result = await (async () => {
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
  })();

  const text = result?.text || cleanOcrText(result?.result?.data?.text || '');
  const lines = (result?.result?.data?.lines || [])
    .map((line) => cleanOcrText(line.text))
    .filter(Boolean);

  return makeOcrResponse({
    lines: lines.length ? lines : text.split(/\n+/),
    confidence: result?.result?.data?.confidence || 0,
    variant: result?.variant || 'unknown',
    provider: 'tesseract'
  });
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

  return makeOcrResponse({
    lines: text.split(/\n+/),
    confidence: 80,
    variant: 'ocr_space_engine_2',
    provider: 'ocr.space'
  });
}

async function getBaiduAccessToken() {
  if (process.env.BAIDU_ACCESS_TOKEN) {
    return process.env.BAIDU_ACCESS_TOKEN;
  }

  const now = Date.now();
  if (baiduTokenCache && baiduTokenCache.expiresAt > now + 60_000) {
    return baiduTokenCache.token;
  }

  const apiKey = process.env.BAIDU_OCR_API_KEY || process.env.BAIDU_API_KEY;
  const secretKey = process.env.BAIDU_OCR_SECRET_KEY || process.env.BAIDU_SECRET_KEY;
  const url = new URL(BAIDU_TOKEN_ENDPOINT);
  url.searchParams.set('grant_type', 'client_credentials');
  url.searchParams.set('client_id', apiKey);
  url.searchParams.set('client_secret', secretKey);

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Accept: 'application/json'
    }
  });

  if (!response.ok) {
    throw new Error(`baidu_token_http_${response.status}`);
  }

  const payload = await response.json();
  if (!payload?.access_token) {
    throw new Error(payload?.error_description || payload?.error || 'baidu_token_missing');
  }

  baiduTokenCache = {
    token: payload.access_token,
    expiresAt: now + Math.max(60, Number(payload.expires_in || 2_592_000) - 300) * 1000
  };
  return baiduTokenCache.token;
}

async function callBaiduOcr(endpoint, buffer, params) {
  const token = await getBaiduAccessToken();
  const url = new URL(endpoint);
  url.searchParams.set('access_token', token);

  const body = new URLSearchParams({
    image: buffer.toString('base64'),
    ...params
  });

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body
  });

  if (!response.ok) {
    throw new Error(`baidu_ocr_http_${response.status}`);
  }

  const payload = await response.json();
  if (payload?.error_code || payload?.error_msg) {
    throw new Error(`baidu_${payload.error_code || 'error'}_${payload.error_msg || ''}`);
  }

  return payload;
}

async function recognizeWithBaidu(buffer) {
  const errors = [];

  try {
    const paperCutPayload = await callBaiduOcr(BAIDU_PAPER_CUT_ENDPOINT, buffer, {
      language_type: 'CHN_ENG',
      detect_direction: 'true',
      words_type: 'handprint_mix',
      splice_text: 'true',
      enhance: 'true',
      only_split: 'false'
    });
    const result = parseBaiduPaperCut(paperCutPayload);
    if (scoreOcrText(result.text) >= 10) {
      return result;
    }
  } catch (error) {
    errors.push(error?.message || String(error));
  }

  try {
    const docPayload = await callBaiduOcr(BAIDU_DOC_ANALYSIS_ENDPOINT, buffer, {
      language_type: 'CHN_ENG',
      result_type: 'big',
      detect_direction: 'true',
      line_probability: 'true',
      words_type: 'handprint_mix',
      layout_analysis: 'true',
      recg_formula: 'true',
      recg_long_division: 'true'
    });
    return parseBaiduDocAnalysis(docPayload);
  } catch (error) {
    errors.push(error?.message || String(error));
  }

  throw new Error(errors.join('; ') || 'baidu_ocr_failed');
}

function parseBaiduPaperCut(payload) {
  const questions = [];
  const lines = [];

  for (const question of payload?.qus_result || []) {
    const elemText = question?.elem_text || {};
    const structuredText = [
      elemText.stem_text,
      elemText.subqus_text,
      elemText.option_text,
      elemText.answer_text,
      elemText.interpretation_text
    ].filter(Boolean);

    const elementLines = [];
    for (const element of question?.qus_element || []) {
      for (const word of element?.elem_word || []) {
        if (word?.word) {
          elementLines.push(word.word);
        }
      }
    }

    const questionLines = compactLines(structuredText.length ? structuredText : elementLines);
    if (questionLines.length) {
      lines.push(...questionLines);
      questions.push({
        type: question.qus_type,
        confidence: question.qus_probability,
        text: questionLines.join('\n')
      });
    }
  }

  return makeOcrResponse({
    lines: lines.length ? lines : extractTextLines(payload),
    confidence: 90,
    variant: 'baidu_paper_cut_edu',
    provider: 'baidu',
    extra: {
      questions,
      baidu_log_id: payload?.log_id
    }
  });
}

function parseBaiduDocAnalysis(payload) {
  return makeOcrResponse({
    lines: extractTextLines(payload),
    confidence: 88,
    variant: 'baidu_doc_analysis',
    provider: 'baidu',
    extra: {
      baidu_log_id: payload?.log_id
    }
  });
}

function extractTextLines(value, lines = []) {
  if (Array.isArray(value)) {
    for (const item of value) {
      extractTextLines(item, lines);
    }
    return lines;
  }

  if (!value || typeof value !== 'object') {
    return lines;
  }

  for (const [key, child] of Object.entries(value)) {
    if (
      typeof child === 'string' &&
      /^(word|words|text|content|elem_text|stem_text|option_text|answer_text|interpretation_text)$/i.test(key)
    ) {
      lines.push(child);
    } else if (typeof child === 'object') {
      extractTextLines(child, lines);
    }
  }

  return lines;
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
    endpoints: ['/api/ocr/recognize', '/ocr/recognize', '/vision/ocr', '/ai/ocr'],
    providers: ['baidu', 'ocr.space', 'tesseract']
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
