document.addEventListener('DOMContentLoaded', function() {
    // Webcam functionality
    const webcamButton = document.getElementById('open-webcam');
    const webcamContainer = document.getElementById('webcam-container');
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('webcam-canvas');
    const captureButton = document.getElementById('capture-button');
    const closeWebcamButton = document.getElementById('close-webcam');
    const webcamResult = document.getElementById('webcam-result');
    const capturedImage = document.getElementById('captured-image');
    const classifyCaptureButton = document.getElementById('classify-capture');
    
    let stream = null;
    
    // Open webcam when button is clicked
    if (webcamButton) {
        webcamButton.addEventListener('click', function() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        facingMode: 'environment',
                        width: { ideal: 1280 },
                        height: { ideal: 720 } 
                    } 
                })
                .then(function(mediaStream) {
                    stream = mediaStream;
                    webcamElement.srcObject = mediaStream;
                    webcamElement.onloadedmetadata = function() {
                        webcamElement.play();
                    };
                    webcamContainer.style.display = 'block';
                    webcamButton.style.display = 'none';
                    
                    // Show a notification
                    const notification = document.createElement('div');
                    notification.className = 'webcam-notification';
                    notification.innerHTML = '<i class="fas fa-camera"></i> Camera activated. Point at waste item and capture.';
                    webcamContainer.appendChild(notification);
                    
                    // Auto-remove notification after 3 seconds
                    setTimeout(() => {
                        notification.classList.add('fade-out');
                        setTimeout(() => notification.remove(), 500);
                    }, 3000);
                })
                .catch(function(err) {
                    console.log("Error accessing webcam: " + err);
                    alert("Error accessing webcam. Please make sure you have granted camera permissions.");
                });
            } else {
                alert("Sorry, your browser doesn't support webcam access.");
            }
        });
    }
    
    // Close webcam
    if (closeWebcamButton) {
        closeWebcamButton.addEventListener('click', function() {
            if (stream) {
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                webcamElement.srcObject = null;
                stream = null;
            }
            webcamContainer.style.display = 'none';
            webcamButton.style.display = 'block';
            webcamResult.style.display = 'none';
        });
    }
    
    // Capture image from webcam
    if (captureButton) {
        captureButton.addEventListener('click', function() {
            // Play capture sound
            const captureSound = new Audio('/static/sounds/camera-shutter.mp3');
            captureSound.play().catch(e => console.log('Sound play failed:', e));
            
            // Add flash effect
            const flash = document.createElement('div');
            flash.className = 'webcam-flash';
            webcamContainer.appendChild(flash);
            
            setTimeout(() => {
                flash.remove();
                
                // Capture the image
                const context = canvasElement.getContext('2d');
                canvasElement.width = webcamElement.videoWidth;
                canvasElement.height = webcamElement.videoHeight;
                context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
                
                // Convert to data URL and display captured image
                const imageDataURL = canvasElement.toDataURL('image/png');
                capturedImage.src = imageDataURL;
                webcamResult.style.display = 'block';
                
                // Copy model selection from main form
                const originalModelOptions = document.querySelector('.model-options:not(.webcam-model-options)');
                const webcamModelOptions = document.querySelector('.webcam-model-options');
                
                if (originalModelOptions && webcamModelOptions) {
                    webcamModelOptions.innerHTML = originalModelOptions.innerHTML;
                }
                
                // Show success message
                const successMsg = document.createElement('div');
                successMsg.className = 'capture-success';
                successMsg.innerHTML = '<i class="fas fa-check-circle"></i> Image captured successfully!';
                webcamResult.prepend(successMsg);
                
                setTimeout(() => {
                    successMsg.classList.add('fade-out');
                    setTimeout(() => successMsg.remove(), 500);
                }, 2000);
            }, 100);
        });
    }
    
    // Classify captured image
    if (classifyCaptureButton) {
        classifyCaptureButton.addEventListener('click', function() {
            // Get selected model
            const selectedModel = document.querySelector('.webcam-model-options input[type="radio"]:checked');
            
            if (!selectedModel) {
                alert("Please select a model first");
                return;
            }
            
            // Convert canvas to blob
            canvasElement.toBlob(function(blob) {
                const formData = new FormData();
                formData.append('file', blob, 'webcam-capture.png');
                formData.append('model', selectedModel.value);
                
                // Show loading overlay
                document.getElementById('loading-overlay').classList.add('active');
                
                // Send to server for classification
                fetch('/classify_image', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading overlay
                    document.getElementById('loading-overlay').classList.remove('active');
                    
                    if (data.error) {
                        alert("Error: " + data.error);
                        return;
                    }
                    
                    // Display results
                    window.location.href = '/?result=webcam&class=' + encodeURIComponent(data.class) + 
                                         '&confidence=' + encodeURIComponent(data.confidence) + 
                                         '&model=' + encodeURIComponent(data.model) +
                                         '&image=' + encodeURIComponent(data.image_path);
                })
                .catch(error => {
                    document.getElementById('loading-overlay').classList.remove('active');
                    console.error('Error:', error);
                    alert("An error occurred during classification.");
                });
            });
        });
    }
    
    // Add camera flash and capture sound effect styles
    const style = document.createElement('style');
    style.textContent = `
        .webcam-flash {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
            opacity: 0.8;
            z-index: 10;
            animation: flash 0.5s ease-out;
        }
        
        @keyframes flash {
            0% { opacity: 0.8; }
            100% { opacity: 0; }
        }
        
        .webcam-notification {
            position: absolute;
            top: 10px;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 0.9rem;
            border-radius: 4px;
            margin: 0 auto;
            width: 80%;
            z-index: 5;
            animation: fadeIn 0.5s ease;
        }
        
        .webcam-notification i {
            margin-right: 5px;
            color: #2ecc71;
        }
        
        .capture-success {
            padding: 10px;
            background: rgba(46, 204, 113, 0.1);
            border-left: 4px solid #2ecc71;
            color: #2ecc71;
            border-radius: 4px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            animation: fadeIn 0.5s ease;
        }
        
        .capture-success i {
            margin-right: 8px;
            font-size: 1.2rem;
        }
        
        .fade-out {
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    `;
    document.head.appendChild(style);
});
