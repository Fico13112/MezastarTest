let mediaRecorder;
let audioChunks = [];
let model;
const labels = ['Zeraora','Solgaleo','Lugia','Ho-Oh','Lunala','Greninja','Eternatus','Keldeo','Grimmsnarl','Zygarde'];

const recordBtn = document.getElementById('recordBtn');
const statusDiv = document.getElementById('status');
const predictionDiv = document.getElementById('prediction');

// Load the model
async function loadModel() {
    statusDiv.innerText = "Status: Loading model...";
    model = await tf.loadLayersModel('tfjs_model/model.json');
    statusDiv.innerText = "Status: Model loaded. Ready!";
}
loadModel();

// Approximate MFCC extraction (match training shape)
async function extractMFCC(audioBuffer) {
    const raw = audioBuffer.getChannelData(0);
    const norm = raw.map(v => v / Math.max(...raw.map(Math.abs)));
    const len = 300;
    const mfcc = new Array(40).fill(0).map(() => new Array(len).fill(0));
    for (let i = 0; i < 40; i++) {
        for (let j = 0; j < len; j++) {
            mfcc[i][j] = norm[Math.floor(j * norm.length / len)] || 0;
        }
    }
    return tf.tensor(mfcc).expandDims(0).expandDims(-1);
}

// Recording and prediction
recordBtn.addEventListener('click', async () => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", e => {
            audioChunks.push(e.data);
        });

        mediaRecorder.addEventListener("stop", async () => {
            predictionDiv.innerText = "Prediction: Processing...";
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

            const X = await extractMFCC(audioBuffer);
            const pred = model.predict(X);
            const idx = pred.argMax(-1).dataSync()[0];
            predictionDiv.innerText = `Prediction: ${labels[idx]}`;
        });

        mediaRecorder.start();
        recordBtn.innerText = "⏹ Stop Recording";
        recordBtn.style.backgroundColor = "#ff6b6b";
    } else if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.innerText = "🎤 Record Sound";
        recordBtn.style.backgroundColor = "#ffcb05";
    }
});


