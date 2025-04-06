import React, { useState, useRef } from 'react';
import { Upload, Camera, AlertCircle, Loader2, ChevronUp, ChevronDown } from 'lucide-react';

type ConversionStatus = 'idle' | 'loading' | 'success' | 'error';

interface ErrorState {
  message: string;
  type: 'camera' | 'upload' | 'api' | null;
}

interface GridData {
  corners: Array<{
    x: number;
    y: number;
    type: string;
  }>;
  cells: Array<{
    id: number;
    x: number;
    y: number;
    element: string;  // Will contain the detected letter/number or "none"
  }>;
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
  const [gridData, setGridData] = useState<GridData | null>(null);
  const [numCols, setNumCols] = useState<number | ''>('');
  const [numRows, setNumRows] = useState<number | ''>('');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  
  // Add after your existing functions, before the startWebcam function
const getCameras = async () => {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    setCameras(videoDevices);
    if (videoDevices.length > 0) {
      setSelectedCamera(videoDevices[0].deviceId);
    }
  } catch (err) {
    setError({ message: 'Unable to get cameras', type: 'camera' });
  }
};

  const handleColsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setNumCols(value === '' ? '' : Math.max(1, parseInt(value) || 1));
  };

  const handleRowsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setNumRows(value === '' ? '' : Math.max(1, parseInt(value) || 1));
  };

  // Replace your existing startWebcam function
const startWebcam = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setShowWebcam(true);
      setError({ message: '', type: null });
    }
  } catch (err) {
    setError({ message: 'Unable to access camera', type: 'camera' });
    console.error('Camera error:', err);
  }
};

  const captureImage = async () => {
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
        await processImage(imageData);
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

    // Find the existing processImage function and replace it with this:
    const processImage = async (imageData: string) => {
      if (numCols === '' || numRows === '') {
        setError({ 
          message: 'Please enter both columns and rows', 
          type: 'api' 
        });
        return;
      }
  
      setStatus('loading');
      try {
        const response = await fetch(
          `http://localhost:5000/process-grid?cols=${numCols}&rows=${numRows}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
          }
        );
  
        if (!response.ok) {
          throw new Error('Failed to process image');
        }
  
        const result = await response.json();
        setGridData(result);
        setStatus('success');
        setError({ message: '', type: null });
        setIsUploadPanelOpen(false);
      } catch (err) {
        setStatus('error');
        setError({ 
          message: err instanceof Error ? err.message : 'Failed to process image', 
          type: 'api' 
        });
      }
    };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.match('image/(jpeg|png)')) {
        setError({ message: 'Please upload a valid image file (.jpg or .png)', type: 'upload' });
        return;
      }

      if (numCols === '' || numRows === '') {
        setError({ 
          message: 'Please enter both columns and rows', 
          type: 'api' 
        });
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      setStatus('loading');
      try {
        const response = await fetch(`http://localhost:5000/process-grid?cols=${numCols}&rows=${numRows}`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Failed to process image');
        }

        const result = await response.json();
        setGridData(result);
        setStatus('success');
        setError({ message: '', type: null });
        
        const reader = new FileReader();
        reader.onload = (e) => {
          setSelectedImage(e.target?.result as string);
        };
        reader.readAsDataURL(file);
      } catch (err) {
        setStatus('error');
        setError({ 
          message: err instanceof Error ? err.message : 'Failed to process image', 
          type: 'api' 
        });
      }
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;
    await processImage(selectedImage);
  };

  return (
    <div className="min-h-screen bg-slate-100">
      <div className="relative">
        {/* Main View Area */}
        <div className="w-full min-h-screen overflow-hidden">
          {status === 'success' && gridData ? (
            <div className="w-full h-full min-h-screen bg-slate-800 rounded-lg p-4 pb-16">
              <h2 className="text-white mb-4">Processed Grid Data:</h2>
              <pre className="text-slate-300 overflow-auto h-[calc(100vh-12rem)]">
                {JSON.stringify(gridData, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="w-full h-full min-h-screen flex items-center justify-center bg-slate-200 text-slate-600">
              Upload or capture a layout to process
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
                {/* Grid Division Inputs */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Columns</label>
                    <input
                      type="number"
                      min="1"
                      value={numCols}
                      onChange={handleColsChange}
                      placeholder="Enter columns"
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Rows</label>
                    <input
                      type="number"
                      min="1"
                      value={numRows}
                      onChange={handleRowsChange}
                      placeholder="Enter rows"
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                </div>

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
                    
                    {/* Add the camera selector here, right after the video element */}
                    {cameras.length > 1 && (
                      <select
                        value={selectedCamera}
                        onChange={(e) => {
                          setSelectedCamera(e.target.value);
                          stopWebcam();
                          startWebcam();
                    }}
                    className="mt-2 w-full p-2 rounded-lg border border-gray-300"
                    >
                    {cameras.map((camera) => (
                      <option key={camera.deviceId} value={camera.deviceId}>
                        {camera.label || `Camera ${camera.deviceId}`}
                      </option>
                  ))}
            </select>
          )}

                    
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
                    'Process Image'
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