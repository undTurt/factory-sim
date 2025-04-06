const API_URL = 'http://localhost:5000';

export const uploadGridImage = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/process-grid`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to process image');
        }

        return await response.json();
    } catch (error) {
        console.error('Error uploading grid image:', error);
        throw error;
    }
};