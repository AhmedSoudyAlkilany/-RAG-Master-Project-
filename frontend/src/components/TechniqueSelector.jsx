import { Database, Search, RefreshCcw, Globe, Zap } from 'lucide-react';

/**
 * Technique Selector Component
 * Dropdown for selecting RAG technique
 */
export default function TechniqueSelector({ value, onChange }) {
    const techniques = [
        {
            id: 'naive',
            name: 'Naive RAG',
            icon: Database,
            color: 'var(--naive-color)',
            description: 'Basic retrieval + generation'
        },
        {
            id: 'hybrid',
            name: 'Hybrid',
            icon: Search,
            color: 'var(--hybrid-color)',
            description: 'BM25 + Vector fusion'
        },
        {
            id: 'rerank',
            name: 'Re-ranking',
            icon: RefreshCcw,
            color: 'var(--rerank-color)',
            description: 'LLM-based re-ranking'
        },
        {
            id: 'crag',
            name: 'Corrective RAG',
            icon: Globe,
            color: 'var(--crag-color)',
            description: 'Self-correcting with web fallback'
        },
    ];

    return (
        <div className="technique-selector">
            {techniques.map((tech) => {
                const Icon = tech.icon;
                const isActive = value === tech.id;

                return (
                    <button
                        key={tech.id}
                        className={`technique-option ${isActive ? 'active' : ''}`}
                        onClick={() => onChange(tech.id)}
                        style={{ '--tech-color': tech.color }}
                    >
                        <div className="tech-icon">
                            <Icon size={18} />
                        </div>
                        <div className="tech-info">
                            <div className="tech-name">{tech.name}</div>
                            <div className="tech-desc">{tech.description}</div>
                        </div>
                        {isActive && <Zap size={16} className="active-indicator" />}
                    </button>
                );
            })}

            <style>{`
        .technique-selector {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .technique-option {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem;
          background: var(--bg-tertiary);
          border: 1px solid transparent;
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--transition-fast);
          text-align: left;
          width: 100%;
        }
        
        .technique-option:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: var(--glass-border);
        }
        
        .technique-option.active {
          background: rgba(99, 102, 241, 0.1);
          border-color: var(--tech-color);
        }
        
        .tech-icon {
          width: 32px;
          height: 32px;
          border-radius: var(--radius-sm);
          background: rgba(255, 255, 255, 0.05);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--tech-color);
          flex-shrink: 0;
        }
        
        .technique-option.active .tech-icon {
          background: var(--tech-color);
          color: white;
        }
        
        .tech-info {
          flex: 1;
          min-width: 0;
        }
        
        .tech-name {
          font-weight: 500;
          font-size: 0.9rem;
          color: var(--text-primary);
        }
        
        .tech-desc {
          font-size: 0.75rem;
          color: var(--text-muted);
        }
        
        .active-indicator {
          color: var(--tech-color);
        }
      `}</style>
        </div>
    );
}
