const etiquetas = ["10 soles", "20 soles", "50 soles", "100 soles"];
let session;

async function cargarModelo() {
  document.getElementById("resultado").innerText = "Cargando modelo ONNX...";
  session = await ort.InferenceSession.create("billetes_model.onnx");
  document.getElementById("resultado").innerText = "✅ Modelo cargado. Activando cámara...";
  iniciarCamara();
}

function iniciarCamara() {
  const video = document.getElementById("video");
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      procesarVideo();
    })
    .catch((err) => {
      console.error("Error al acceder a la cámara:", err);
    });
}

async function procesarVideo() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  setInterval(async () => {
    ctx.drawImage(video, 0, 0, 150, 150);
    const imageData = ctx.getImageData(0, 0, 150, 150);

    // Preprocesar imagen
    const floatData = new Float32Array(150 * 150 * 3);
    for (let i = 0; i < imageData.data.length / 4; i++) {
      floatData[i * 3 + 0] = imageData.data[i * 4 + 0] / 255;
      floatData[i * 3 + 1] = imageData.data[i * 4 + 1] / 255;
      floatData[i * 3 + 2] = imageData.data[i * 4 + 2] / 255;
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, 150, 150]);
    const results = await session.run({ input: tensor });
    const output = results.output.data;
    const idx = output.indexOf(Math.max(...output));

    document.getElementById("resultado").innerText = `💵 Billete detectado: ${etiquetas[idx]}`;
  }, 1000);
}

cargarModelo();
