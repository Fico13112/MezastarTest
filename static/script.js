const statusDiv = document.getElementById('status');
const predictionDiv = document.getElementById('prediction');
const recordBtn = document.getElementById('record');

recordBtn.addEventListener('click', async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser does not support audio recording.");
        return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    let chunks = [];

    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', blob, 'sound.wav');

        statusDiv.innerText = "Status: Sending audio for prediction...";
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (data.prediction) {
            predictionDiv.innerText = `Prediction: ${data.prediction} (Confidence: ${(data.confidence*100).toFixed(2)}%)`;
            statusDiv.innerText = "Status: Ready";
        } else {
            predictionDiv.innerText = "Prediction failed.";
            statusDiv.innerText = "Status: Error";
        }
    };

    mediaRecorder.start();
    statusDiv.innerText = "Status: Recording 3 seconds...";
    setTimeout(() => mediaRecorder.stop(), 3000);
});
