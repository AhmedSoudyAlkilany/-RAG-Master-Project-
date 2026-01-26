import { useState, useEffect } from 'react';
import { Cpu, Box, Loader2 } from 'lucide-react';
import { getModels, switchModels } from '../services/api';

/**
 * Model Selector Component
 * Dropdown for switching Ollama models
 */
export default function ModelSelector() {
    const [models, setModels] = useState(null);
    const [loading, setLoading] = useState(false);
    const [switching, setSwitching] = useState(false);

    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        setLoading(true);
        try {
            const result = await getModels();
            setModels(result);
        } catch (error) {
            console.error('Failed to load models:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSwitch = async (type, model) => {
        setSwitching(true);
        try {
            const result = await switchModels(
                type === 'llm' ? model : null,
                type === 'embedding' ? model : null
            );
            setModels(prev => ({
                ...prev,
                active_llm: result.active_llm,
                active_embedding: result.active_embedding
            }));
        } catch (error) {
            console.error('Failed to switch model:', error);
        } finally {
            setSwitching(false);
        }
    };

    if (loading || !models) {
        return (
            <div className="model-loading">
                <Loader2 className="animate-spin" size={20} />
                <span>Loading models...</span>
            </div>
        );
    }

    return (
        <div className="model-selector">
            {/* LLM Model */}
            <div className="model-group">
                <label className="model-label">
                    <Cpu size={14} /> LLM Model
                </label>
                <select
                    className="input select"
                    value={models.active_llm}
                    onChange={(e) => handleSwitch('llm', e.target.value)}
                    disabled={switching}
                >
                    {models.llm_models.map(model => (
                        <option key={model} value={model}>{model}</option>
                    ))}
                </select>
            </div>

            {/* Embedding Model */}
            <div className="model-group">
                <label className="model-label">
                    <Box size={14} /> Embedding Model
                </label>
                <select
                    className="input select"
                    value={models.active_embedding}
                    onChange={(e) => handleSwitch('embedding', e.target.value)}
                    disabled={switching}
                >
                    {models.embedding_models.map(model => (
                        <option key={model} value={model}>{model}</option>
                    ))}
                </select>
            </div>

            <style>{`
        .model-selector {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        
        .model-loading {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: var(--text-muted);
          padding: 1rem 0;
        }
        
        .model-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .model-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.8rem;
          color: var(--text-secondary);
        }
        
        .model-selector .select {
          font-size: 0.85rem;
          padding: 0.625rem 2rem 0.625rem 0.75rem;
        }
      `}</style>
        </div>
    );
}
