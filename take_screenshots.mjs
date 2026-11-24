import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const SCREENSHOT_DIR = path.resolve('screenshots');
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

// Bersihkan direktori screenshots
for (const file of fs.readdirSync(SCREENSHOT_DIR)) {
  if (file.endsWith('.png')) {
    fs.unlinkSync(path.join(SCREENSHOT_DIR, file));
  }
}

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const TEMP_PROFILE = path.resolve(process.env.TEMP || 'C:\\Windows\\Temp', 'edge_fullpage_profile_' + Date.now());
const PORT = 9222;

console.log('Menjalankan Microsoft Edge Desktop (1920x1080)...');
const edgeProc = spawn(EDGE_PATH, [
  `--remote-debugging-port=${PORT}`,
  '--headless=new',
  '--hide-scrollbars',
  '--no-first-run',
  '--no-default-browser-check',
  `--user-data-dir=${TEMP_PROFILE}`,
  '--window-size=1920,1080',
  'about:blank'
], { stdio: 'ignore', detached: false });

const sleep = (ms) => new Promise(res => setTimeout(res, ms));

async function waitForEndpoint(maxTries = 30) {
  for (let i = 0; i < maxTries; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json`);
      if (res.ok) {
        const list = await res.json();
        const page = list.find(item => item.type === 'page');
        if (page && page.webSocketDebuggerUrl) {
          return page.webSocketDebuggerUrl;
        }
      }
    } catch (e) {}
    await sleep(300);
  }
  throw new Error('Gagal menghubungkan ke Edge remote debugging endpoint');
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.msgId = 1;
    this.callbacks = new Map();
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.id && this.callbacks.has(data.id)) {
            const { resolve, reject } = this.callbacks.get(data.id);
            this.callbacks.delete(data.id);
            if (data.error) reject(new Error(data.error.message));
            else resolve(data.result);
          }
        } catch (e) {
          console.error('Error parsing CDP message:', e);
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.msgId++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expr) {
    const res = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    return res.result?.value;
  }

  async clickMouse(x, y) {
    await this.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: Math.round(x), y: Math.round(y), button: 'left', clickCount: 1 });
    await this.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: Math.round(x), y: Math.round(y), button: 'left', clickCount: 1 });
  }

  async waitForStreamlitReady(timeoutMs = 15000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const ready = await this.eval(`
        (() => {
          const root = document.querySelector('div[data-testid="stAppViewContainer"]');
          const running = document.querySelector('div[data-testid="stStatusWidget"]');
          return !!root && !running;
        })()
      `);
      if (ready) {
        await sleep(1500);
        return true;
      }
      await sleep(300);
    }
    return false;
  }

  async setSliderByRatio(sliderIndex, ratio) {
    const box = await this.eval(`
      (() => {
        const sliders = Array.from(document.querySelectorAll('div[data-testid="stSlider"]'));
        if (!sliders[${sliderIndex}]) return null;
        const track = sliders[${sliderIndex}].querySelector('div[data-orientation="horizontal"]');
        if (!track) return null;
        const r = track.getBoundingClientRect();
        return { x: r.x + r.width * ${ratio}, y: r.y + r.height/2 };
      })()
    `);
    if (box) {
      await this.clickMouse(box.x, box.y);
      await sleep(1500);
    }
  }

  async setSelectboxOption(selectIndex, targetOptionText) {
    const btnBox = await this.eval(`
      (() => {
        const selectboxes = document.querySelectorAll('div[data-testid="stSelectbox"]');
        if (!selectboxes[${selectIndex}]) return null;
        const btn = selectboxes[${selectIndex}].querySelector('button[aria-label="Open"]');
        if (!btn) return null;
        const r = btn.getBoundingClientRect();
        return { x: r.x + r.width/2, y: r.y + r.height/2 };
      })()
    `);
    if (btnBox) {
      await this.clickMouse(btnBox.x, btnBox.y);
      await sleep(500);

      const optBox = await this.eval(`
        (() => {
          const opts = Array.from(document.querySelectorAll('[role="option"]'));
          const target = opts.find(o => o.innerText.trim() === '${targetOptionText}');
          if (!target) return null;
          const r = target.getBoundingClientRect();
          return { x: r.x + r.width/2, y: r.y + r.height/2 };
        })()
      `);

      if (optBox) {
        await this.clickMouse(optBox.x, optBox.y);
      } else {
        await this.clickMouse(btnBox.x, btnBox.y);
      }
      await sleep(1500);
    }
  }

  async applyPerfectFullPageLayout() {
    await this.eval(`
      (() => {
        let style = document.getElementById('fullpage-perfect-style');
        if (!style) {
          style = document.createElement('style');
          style.id = 'fullpage-perfect-style';
          document.head.appendChild(style);
        }
        style.innerHTML = \`
          /* Kunci html, body, dan appView agar tidak memotong ke bawah dan tidak geser ke samping */
          html, body, #root, .stApp, [data-testid="stAppViewContainer"] {
            height: auto !important;
            min-height: 100% !important;
            max-height: none !important;
            overflow-x: hidden !important;
            overflow-y: visible !important;
          }
          /* Sidebar tampil utuh tanpa scroll internal dan terkunci di x=0 */
          section[data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"], div[data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
          }
          /* Area konten utama melebar proporsional sehingga tabel 10 kolom tidak terpotong */
          .main, [data-testid="stMainBlockContainer"] {
            max-width: 1500px !important;
            width: calc(100% - 350px) !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
          }
          /* Lebarkan tabel data siswa agar seluruh 10 kolom tampil utuh */
          [data-testid="stDataFrameResizable"] {
            width: 1350px !important;
            max-width: 1350px !important;
          }
          /* Hilangkan header dan footer bawaan agar tampilan profesional */
          header[data-testid="stHeader"], footer {
            display: none !important;
          }
          .stApp {
            padding-bottom: 60px !important;
          }
        \`;

        // Lebarkan canvas dataframe
        document.querySelectorAll('[data-testid="stDataFrameResizable"]').forEach(el => {
          el.style.setProperty('width', '1350px', 'important');
          el.style.setProperty('max-width', '1350px', 'important');
        });
        window.dispatchEvent(new Event('resize'));

        // Reset scroll position ke pojok kiri atas
        window.scrollTo(0, 0);
        const container = document.querySelector('[data-testid="stAppViewContainer"]');
        if (container) {
          container.scrollLeft = 0;
          container.scrollTop = 0;
        }
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
          sidebar.scrollLeft = 0;
          sidebar.scrollTop = 0;
        }
      })()
    `);
    await sleep(1000);
  }

  async captureFullPage(filename) {
    console.log(`Mengambil screenshot full-page rapi: ${filename}...`);
    
    // Terapkan layout sempurna
    await this.applyPerfectFullPageLayout();

    // Hitung tinggi dokumen total
    const metrics = await this.eval(`
      (() => {
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        const main = document.querySelector('[data-testid="stMainBlockContainer"]') || document.querySelector('.main');
        const sH = sidebar ? sidebar.scrollHeight : 0;
        const mH = main ? main.scrollHeight : 0;
        return {
          width: 1920,
          height: Math.max(sH, mH, 1250) + 70
        };
      })()
    `);

    // Override viewport ke seluruh tinggi halaman
    await this.send('Emulation.setDeviceMetricsOverride', {
      width: metrics.width,
      height: metrics.height,
      deviceScaleFactor: 1,
      mobile: false
    });

    await sleep(1500);

    // Capture screenshot dari viewport (captureBeyondViewport: false untuk menjamin x=0 y=0)
    const screenshot = await this.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: false
    });

    const filePath = path.join(SCREENSHOT_DIR, filename);
    fs.writeFileSync(filePath, Buffer.from(screenshot.data, 'base64'));
    const sizeKB = (fs.statSync(filePath).size / 1024).toFixed(1);
    console.log(`✓ Tersimpan: ${filename} (${metrics.width}x${metrics.height} px, ${sizeKB} KB)`);

    // Kembalikan viewport standar untuk interaksi berikutnya
    await this.send('Emulation.setDeviceMetricsOverride', {
      width: 1920,
      height: 1080,
      deviceScaleFactor: 1,
      mobile: false
    });
    await sleep(500);
  }

  close() {
    this.ws.close();
  }
}

async function run() {
  try {
    const wsUrl = await waitForEndpoint();
    console.log('Terhubung ke WebSocket CDP:', wsUrl);

    const client = new CDPClient(wsUrl);
    await client.connect();

    await client.send('Page.enable');
    await client.send('DOM.enable');
    await client.send('Runtime.enable');

    // Kunci desktop 1920x1080 SEJAK AWAL
    await client.send('Emulation.setDeviceMetricsOverride', {
      width: 1920,
      height: 1080,
      deviceScaleFactor: 1,
      mobile: false
    });

    const appUrl = 'http://localhost:8501';
    console.log(`Membuka aplikasi Streamlit: ${appUrl}...`);
    await client.send('Page.navigate', { url: appUrl });
    await client.waitForStreamlitReady();

    // =========================================================================
    // SKENARIO 1: Simulasi Siswa Prioritas Penerima KIP
    // - Penghasilan Orang Tua: Rp 1.200.000 (12% track slider)
    // - Peserta KPS: "Ya"
    // - Alat Transportasi: "Jalan kaki"
    // - Jumlah Saudara Kandung: 3 (60% track slider)
    // - Jarak Rumah ke Sekolah: 4 km (80% track slider)
    // - Rekomendasi BSM: "Ya"
    // =========================================================================
    console.log('\n--- Menjalankan Simulasi 1: Siswa Prioritas Penerima KIP ---');
    await client.setSliderByRatio(0, 0.12);
    await client.setSelectboxOption(0, 'Ya');
    await client.setSelectboxOption(1, 'Jalan kaki');
    await client.setSliderByRatio(1, 0.60);
    await client.setSliderByRatio(2, 0.80);
    await client.setSelectboxOption(2, 'Ya');

    await client.waitForStreamlitReady();
    await sleep(2500);

    await client.captureFullPage('01_simulasi_siswa_prioritas_penerima_kip.png');

    // =========================================================================
    // SKENARIO 2: Simulasi Siswa Keluarga Mampu (Non-KIP)
    // - Penghasilan Orang Tua: Rp 4.500.000 (45% track slider)
    // - Peserta KPS: "Tidak"
    // - Alat Transportasi: "Kendaraan pribadi"
    // - Jumlah Saudara Kandung: 1 (20% track slider)
    // - Jarak Rumah ke Sekolah: 1 km (20% track slider)
    // - Rekomendasi BSM: "Tidak"
    // =========================================================================
    console.log('\n--- Menjalankan Simulasi 2: Siswa Keluarga Mampu (Non-KIP) ---');
    await client.setSliderByRatio(0, 0.45);
    await client.setSelectboxOption(0, 'Tidak');
    await client.setSelectboxOption(1, 'Kendaraan pribadi');
    await client.setSliderByRatio(1, 0.20);
    await client.setSliderByRatio(2, 0.20);
    await client.setSelectboxOption(2, 'Tidak');

    await client.waitForStreamlitReady();
    await sleep(2500);

    await client.captureFullPage('02_simulasi_siswa_keluarga_mampu_non_kip.png');

    // =========================================================================
    // SKENARIO 3: Simulasi Batch Upload File CSV (1.csv)
    // =========================================================================
    console.log('\n--- Menjalankan Simulasi 3: Inferensi Menggunakan Upload CSV (1.csv) ---');
    const doc = await client.send('DOM.getDocument');
    const fileInputNode = await client.send('DOM.querySelector', {
      nodeId: doc.root.nodeId,
      selector: 'input[type="file"]'
    });

    if (fileInputNode && fileInputNode.nodeId) {
      const csvPath = path.resolve('streamlit', '1.csv');
      console.log(`Mengunggah file CSV: ${csvPath}`);
      await client.send('DOM.setFileInputFiles', {
        files: [csvPath],
        nodeId: fileInputNode.nodeId
      });

      console.log('Menunggu pemrosesan inferensi file CSV...');
      await client.waitForStreamlitReady(15000);
      await sleep(3000);

      await client.captureFullPage('03_simulasi_inferensi_upload_csv.png');
    }

    client.close();
    console.log('\nSemua screenshot full page simulasi berhasil diambil secara sempurna, rapi, dan tanpa terpotong!');
  } catch (err) {
    console.error('Error saat capture screenshot:', err);
  } finally {
    try {
      edgeProc.kill();
    } catch (e) {}
  }
}

run();
