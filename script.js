let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById('recordBtn');
const predictionDiv = document.getElementById('prediction');

recordBtn.addEventListener('click', async () => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", e => {
            audioChunks.push(e.data);
        });

        mediaRecorder.addEventListener("stop", async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('file', audioBlob, 'recorded.wav');

            predictionDiv.innerText = "Predicting...";

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                predictionDiv.innerText = `Prediction: ${data.prediction}`;
            } catch (err) {
                predictionDiv.innerText = "Error: Could not get prediction.";
                console.error(err);
            }
        });

        mediaRecorder.start();
        recordBtn.innerText = "⏹ Stop Recording";
        recordBtn.style.backgroundColor = "#ff6b6b";
    } else if (mediaRecorder.state === "recording") {
        // Stop recording
        mediaRecorder.stop();
        recordBtn.innerText = "🎤 Record Sound";
        recordBtn.style.backgroundColor = "#ffcb05";
    }
});
