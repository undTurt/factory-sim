import { useState } from 'react';
import { uploadGridImage } from '../services/api';

export const FileUpload = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [gridData, setGridData] = useState(null);

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const result = await uploadGridImage(file);
            setGridData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to process image');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4">
            <input 
                type="file" 
                accept="image/*"
                onChange={handleFileUpload}
                className="mb-4"
                disabled={loading}
            />
            
            {loading && <p>Processing image...</p>}
            {error && <p className="text-red-500">{error}</p>}
            
            {gridData && (
                <div className="mt-4">
                    <h3>Grid Data:</h3>
                    <pre>{JSON.stringify(gridData, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};