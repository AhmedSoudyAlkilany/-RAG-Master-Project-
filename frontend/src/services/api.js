/**
 * RAG Master Project - API Service
 * Handles all backend communication
 */

const API_BASE = '/api';

/**
 * Query the RAG system
 */
export async function queryRAG(question, technique = 'hybrid', options = {}) {
    const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question,
            technique,
            return_sources: true,
            hybrid_alpha: options.hybridAlpha,
            reranker_type: options.rerankerType,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }

    return response.json();
}

/**
 * Compare multiple RAG techniques
 */
export async function compareTechniques(question, techniques = ['naive', 'hybrid', 'rerank', 'crag']) {
    const response = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, techniques }),
    });

    if (!response.ok) {
        throw new Error('Comparison failed');
    }

    return response.json();
}

/**
 * Upload a document
 */
export async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
}

/**
 * List documents
 */
export async function listDocuments() {
    const response = await fetch(`${API_BASE}/documents`);

    if (!response.ok) {
        throw new Error('Failed to list documents');
    }

    return response.json();
}

/**
 * Clear all documents
 */
export async function clearDocuments() {
    const response = await fetch(`${API_BASE}/documents`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        throw new Error('Failed to clear documents');
    }

    return response.json();
}

/**
 * Get available models
 */
export async function getModels() {
    const response = await fetch(`${API_BASE}/models`);

    if (!response.ok) {
        throw new Error('Failed to get models');
    }

    return response.json();
}

/**
 * Switch models
 */
export async function switchModels(llmModel, embeddingModel) {
    const response = await fetch(`${API_BASE}/models/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            llm_model: llmModel,
            embedding_model: embeddingModel,
        }),
    });

    if (!response.ok) {
        throw new Error('Failed to switch models');
    }

    return response.json();
}

/**
 * Health check
 */
export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        return response.json();
    } catch (error) {
        return { status: 'error', ollama_connected: false };
    }
}
