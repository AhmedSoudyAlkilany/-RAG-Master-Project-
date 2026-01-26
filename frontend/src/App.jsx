import { useState, useEffect } from 'react';
import {
    MessageSquare, Upload, Settings, Book,
    Send, Loader2, FileText, Trash2, CheckCircle, AlertCircle
} from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import DocumentUpload from './components/DocumentUpload';
import TechniqueSelector from './components/TechniqueSelector';
import ModelSelector from './components/ModelSelector';
import { checkHealth } from './services/api';

/**
 * RAG Master Project - Main Application
 */
export default function App() {
    const [activeTab, setActiveTab] = useState('chat');
    const [health, setHealth] = useState(null);
    const [technique, setTechnique] = useState('hybrid');

    // Check backend health on mount
    useEffect(() => {
        checkHealth().then(setHealth).catch(() => setHealth({ status: 'error' }));
    }, []);

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="header-content">
                    <div className="logo">
                        <div className="logo-icon">
                            <MessageSquare size={24} />
                        </div>
                        <div>
                            <h1>RAG Master</h1>
                            <p className="tagline">Advanced Retrieval-Augmented Generation</p>
                        </div>
                    </div>

                    <div className="header-right">
                        <div className="health-status">
                            {health?.status === 'healthy' ? (
                                <span className="status-ok"><CheckCircle size={16} /> Connected</span>
                            ) : (
                                <span className="status-error"><AlertCircle size={16} /> Disconnected</span>
                            )}
                        </div>

                        <nav className="nav-tabs">
                            <button
                                className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
                                onClick={() => setActiveTab('chat')}
                            >
                                <MessageSquare size={18} /> Chat
                            </button>
                            <button
                                className={`nav-tab ${activeTab === 'docs' ? 'active' : ''}`}
                                onClick={() => setActiveTab('docs')}
                            >
                                <FileText size={18} /> Documents
                            </button>
                            <button
                                className={`nav-tab ${activeTab === 'learn' ? 'active' : ''}`}
                                onClick={() => setActiveTab('learn')}
                            >
                                <Book size={18} /> Learn
                            </button>
                        </nav>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="main">
                {activeTab === 'chat' && (
                    <div className="chat-layout">
                        <aside className="sidebar">
                            <div className="sidebar-section">
                                <h3>RAG Technique</h3>
                                <TechniqueSelector
                                    value={technique}
                                    onChange={setTechnique}
                                />
                            </div>

                            <div className="sidebar-section">
                                <h3>Models</h3>
                                <ModelSelector />
                            </div>

                            <div className="sidebar-section">
                                <h3>Quick Upload</h3>
                                <DocumentUpload compact />
                            </div>
                        </aside>

                        <div className="chat-container">
                            <ChatInterface technique={technique} />
                        </div>
                    </div>
                )}

                {activeTab === 'docs' && (
                    <div className="docs-page">
                        <h2>Document Management</h2>
                        <p className="text-muted">Upload documents to build your knowledge base</p>
                        <DocumentUpload />
                    </div>
                )}

                {activeTab === 'learn' && (
                    <div className="learn-page">
                        <h2>RAG Techniques Guide</h2>
                        <p className="text-muted">Learn about the different RAG techniques implemented in this project</p>

                        <div className="technique-cards">
                            <TechniqueCard
                                name="Hybrid Retrieval"
                                color="var(--hybrid-color)"
                                description="Combines BM25 keyword search with vector semantic search using Reciprocal Rank Fusion."
                                useCases={['Technical documentation', 'Specialized terminology', 'Multi-domain knowledge']}
                                complexity="Medium"
                            />

                            <TechniqueCard
                                name="Re-ranking"
                                color="var(--rerank-color)"
                                description="Post-processes initial retrieval with LLM or cross-encoder scoring for improved relevance."
                                useCases={['Critical information needs', 'Ambiguous queries', 'High-stakes applications']}
                                complexity="Medium"
                            />

                            <TechniqueCard
                                name="Corrective RAG (CRAG)"
                                color="var(--crag-color)"
                                description="Self-correcting pipeline that evaluates document relevance and falls back to web search."
                                useCases={['Knowledge gaps', 'Real-time information', 'High reliability needs']}
                                complexity="High"
                            />
                        </div>
                    </div>
                )}
            </main>

            <style>{`
        .app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }
        
        /* Header */
        .header {
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--glass-border);
          padding: 1rem 2rem;
          position: sticky;
          top: 0;
          z-index: 100;
          backdrop-filter: blur(12px);
        }
        
        .header-content {
          max-width: 1600px;
          margin: 0 auto;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        
        .logo {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        
        .logo-icon {
          width: 48px;
          height: 48px;
          background: var(--accent-gradient);
          border-radius: var(--radius-md);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }
        
        .logo h1 {
          font-size: 1.5rem;
          font-weight: 700;
          margin: 0;
        }
        
        .tagline {
          font-size: 0.8rem;
          color: var(--text-muted);
          margin: 0;
        }
        
        .header-right {
          display: flex;
          align-items: center;
          gap: 2rem;
        }
        
        .health-status {
          font-size: 0.85rem;
        }
        
        .status-ok { color: var(--success); display: flex; align-items: center; gap: 0.5rem; }
        .status-error { color: var(--error); display: flex; align-items: center; gap: 0.5rem; }
        
        .nav-tabs {
          display: flex;
          gap: 0.5rem;
        }
        
        .nav-tab {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.25rem;
          background: transparent;
          border: 1px solid transparent;
          border-radius: var(--radius-md);
          color: var(--text-secondary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }
        
        .nav-tab:hover {
          background: var(--glass-bg);
          color: var(--text-primary);
        }
        
        .nav-tab.active {
          background: var(--accent-gradient);
          color: white;
          border-color: transparent;
        }
        
        /* Main */
        .main {
          flex: 1;
          padding: 2rem;
          max-width: 1600px;
          margin: 0 auto;
          width: 100%;
        }
        
        /* Chat Layout */
        .chat-layout {
          display: grid;
          grid-template-columns: 300px 1fr;
          gap: 2rem;
          height: calc(100vh - 180px);
        }
        
        .sidebar {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        
        .sidebar-section {
          background: var(--bg-card);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          padding: 1.25rem;
          backdrop-filter: blur(12px);
        }
        
        .sidebar-section h3 {
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 1rem;
        }
        
        .chat-container {
          background: var(--bg-card);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          backdrop-filter: blur(12px);
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        
        /* Docs Page */
        .docs-page, .learn-page {
          max-width: 1200px;
          margin: 0 auto;
        }
        
        .docs-page h2, .learn-page h2 {
          margin-bottom: 0.5rem;
        }
        
        /* Technique Cards */
        .technique-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
          gap: 1.5rem;
          margin-top: 2rem;
        }
        
        .technique-card {
          background: var(--bg-card);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          padding: 1.5rem;
          transition: all var(--transition-normal);
        }
        
        .technique-card:hover {
          transform: translateY(-4px);
          box-shadow: var(--shadow-lg);
        }
        
        .technique-card h3 {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 1rem;
        }
        
        .technique-card h3::before {
          content: '';
          width: 4px;
          height: 24px;
          background: var(--card-color);
          border-radius: 2px;
        }
        
        .technique-card p {
          color: var(--text-secondary);
          margin-bottom: 1rem;
        }
        
        .use-cases {
          list-style: none;
        }
        
        .use-cases li {
          padding: 0.5rem 0;
          border-bottom: 1px solid var(--glass-border);
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .use-cases li:last-child {
          border-bottom: none;
        }
        
        .complexity-badge {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          background: rgba(255,255,255,0.1);
          border-radius: var(--radius-sm);
          font-size: 0.75rem;
          font-weight: 600;
          margin-top: 1rem;
        }
      `}</style>
        </div>
    );
}


// Technique Card Component
function TechniqueCard({ name, color, description, useCases, complexity }) {
    return (
        <div className="technique-card" style={{ '--card-color': color }}>
            <h3>{name}</h3>
            <p>{description}</p>
            <ul className="use-cases">
                {useCases.map((use, i) => (
                    <li key={i}>• {use}</li>
                ))}
            </ul>
            <span className="complexity-badge">Complexity: {complexity}</span>
        </div>
    );
}
