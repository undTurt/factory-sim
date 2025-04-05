import React, { useState, useRef } from 'react';
import { Upload, Camera, AlertCircle, Loader2, ChevronUp, ChevronDown } from 'lucide-react';

type ConversionStatus = 'idle' | 'loading' | 'success' | 'error';

interface ErrorState {
  message: string;
  type: 'camera' | 'upload' | 'api' | null;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [status, setStatus] = useState<ConversionStatus>('idle');
  const [error, setError] = useState<ErrorState>({ message: '', type: null });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showWebcam, setShowWebcam] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isUploadPanelOpen, setIsUploadPanelOpen] = useState(true);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setShowWebcam(true);
        setError({ message: '', type: null });
      }
    } catch (err) {
      setError({ message: 'Unable to access camera', type: 'camera' });
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        const imageData = canvasRef.current.toDataURL('image/jpeg');
        setSelectedImage(imageData);
        setShowWebcam(false);
        stopWebcam();
      }
    }
  };

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setShowWebcam(false);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.match('image/(jpeg|png)')) {
        setError({ message: 'Please upload a valid image file (.jpg or .png)', type: 'upload' });
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        setError({ message: '', type: null });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    setStatus('loading');
    try {
      const response = await fetch('/api/convert-layout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: selectedImage }),
      });

      if (!response.ok) {
        throw new Error('Failed to convert layout');
      }

      const data = await response.json();
      setStatus('success');
      setIsUploadPanelOpen(false);
      // Handle the 3D data response here
      console.log(data);
    } catch (err) {
      setStatus('error');
      setError({ message: 'Failed to process image', type: 'api' });
    }
  };

  return (
    <div className="min-h-screen bg-slate-100">
      <div className="relative">
        {/* Main 3D View Area */}
        <div className={`w-full transition-all duration-300 ${status === 'success' ? 'h-screen' : 'h-[calc(100vh-64px)]'}`}>
          {status === 'success' ? (
            <div className="w-full h-full bg-slate-800 rounded-lg">
              {/* Unity WebGL component would be mounted here */}
              <div className="w-full h-full flex items-center justify-center text-slate-400">
                3D Factory Layout View
              </div>
            </div>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-slate-200 text-slate-600">
              Upload or capture a layout to view the 3D simulation
            </div>
          )}
        </div>

        {/* Collapsible Upload Panel */}
        <div className={`fixed bottom-0 left-0 right-0 bg-white shadow-lg transition-transform duration-300 ${isUploadPanelOpen ? 'translate-y-0' : 'translate-y-[calc(100%-3rem)]'}`}>
          <button
            onClick={() => setIsUploadPanelOpen(!isUploadPanelOpen)}
            className="w-full py-2 px-4 flex items-center justify-between bg-slate-800 text-white"
          >
            <span className="font-semibold">Upload Layout</span>
            {isUploadPanelOpen ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
          </button>

          <div className="p-6">
            <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
              <div className="space-y-4">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Upload size={20} />
                  Upload Image
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  onChange={handleFileUpload}
                  className="hidden"
                />

                <button
                  onClick={showWebcam ? stopWebcam : startWebcam}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition-colors"
                >
                  <Camera size={20} />
                  {showWebcam ? 'Stop Camera' : 'Use Camera'}
                </button>

                {error.message && (
                  <div className="p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
                    <AlertCircle size={20} />
                    {error.message}
                  </div>
                )}

                {showWebcam && (
                  <div className="mt-4">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full rounded-lg"
                    />
                    <button
                      onClick={captureImage}
                      className="mt-2 w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                    >
                      Capture
                    </button>
                  </div>
                )}
              </div>

              <div>
                <div className="aspect-video bg-slate-100 rounded-lg overflow-hidden">
                  {selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Selected layout"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-400">
                      No image selected
                    </div>
                  )}
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!selectedImage || status === 'loading'}
                  className={`mt-4 w-full px-4 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors ${
                    !selectedImage || status === 'loading'
                      ? 'bg-slate-300 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {status === 'loading' ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      Processing...
                    </>
                  ) : (
                    'Convert to 3D'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Hidden canvas for webcam capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}

export default App;