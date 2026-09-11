const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("captureButton");
const statusBox = document.getElementById("status");
 
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        const cookies = document.cookie.split(";");
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith(name + "=")) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });
 
        video.srcObject = stream;
        await video.play();
 
        statusBox.textContent = "カメラ準備完了。正面を向いてください。";
        captureButton.disabled = false;
    } catch (error) {
        console.error(error);
        statusBox.textContent =
            "カメラを起動できません。ブラウザのカメラ権限を確認してください。";
    }
}
 
captureButton.addEventListener("click", async () => {
    captureButton.disabled = true;
    statusBox.textContent = "認証中...";
 
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
 
    const imageData = canvas.toDataURL("image/jpeg", 0.90);
 
    try {
        const response = await fetch(window.RECOGNIZE_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken")
            },
            body: JSON.stringify({ image: imageData })
        });
 
        const data = await response.json();
 
        if (data.ok) {
            statusBox.textContent =
                `${data.student} さん：${data.type} ${data.time}`;
 
            setTimeout(() => {
                window.location.href = data.redirect_url;
            }, 600);
        } else {
            statusBox.textContent = data.message;
            captureButton.disabled = false;
        }
        } catch (error) {
        console.error(error);
        statusBox.textContent = "通信エラーが発生しました。";
        captureButton.disabled = false;
    }
});
 
startCamera();

